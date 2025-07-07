import asyncio
import functools
import inspect
from enum import Enum
from typing import AsyncIterator, Optional

import pytest

from mesh.utils.auth import TokenRSAAuthorizerBase


class AuthRole(Enum):
    CLIENT = 0
    SERVICER = 1

class AuthRPCWrapper:
    def __init__(self, stub, role: AuthRole, authorizer, service_public_key: Optional[bytes] = None):
        self._stub = stub
        self._role = role
        self._authorizer = authorizer
        self._service_public_key = service_public_key

    def __getattribute__(self, name: str):
        if not name.startswith("rpc_"):
            return object.__getattribute__(self, name)

        stub = object.__getattribute__(self, "_stub")
        method = getattr(stub, name)
        role = object.__getattribute__(self, "_role")
        authorizer = object.__getattribute__(self, "_authorizer")
        service_public_key = object.__getattribute__(self, "_service_public_key")

        if inspect.isasyncgenfunction(method):
            # STREAMING RPC
            @functools.wraps(method)
            async def wrapped_stream_rpc(request, *args, **kwargs):
                if authorizer:
                    if role == AuthRole.CLIENT:
                        await authorizer.sign_request(request, service_public_key)
                    elif role == AuthRole.SERVICER:
                        if not await authorizer.validate_request(request):
                            return

                async for response in method(request, *args, **kwargs):
                    if authorizer:
                        if role == AuthRole.SERVICER:
                            await authorizer.sign_response(response, request)
                        elif role == AuthRole.CLIENT:
                            if not await authorizer.validate_response(response, request):
                                continue
                    yield response

            return wrapped_stream_rpc

        else:
            # UNARY RPC
            @functools.wraps(method)
            async def wrapped_unary_rpc(request, *args, **kwargs):
                if authorizer:
                    if role == AuthRole.CLIENT:
                        await authorizer.sign_request(request, service_public_key)
                    elif role == AuthRole.SERVICER:
                        if not await authorizer.validate_request(request):
                            return None

                response = await method(request, *args, **kwargs)

                if authorizer:
                    if role == AuthRole.SERVICER:
                        await authorizer.sign_response(response, request)
                    elif role == AuthRole.CLIENT:
                        if not await authorizer.validate_response(response, request):
                            return None

                return response

            return wrapped_unary_rpc


# Dummy classes
class DummyRequest:
    def __init__(self, msg):
        self.msg = msg
        self.auth = type("auth", (), {"signature": b""})()

    def SerializeToString(self): return self.msg.encode()

class DummyResponse:
    def __init__(self, reply): self.reply = reply
    def SerializeToString(self): return self.reply.encode()
    auth = type("auth", (), {"signature": b""})()

class DummyAuthorizer:
    async def sign_request(self, request, pub): request.auth.signature = b"signed"
    async def validate_request(self, request): return request.auth.signature == b"signed"
    async def sign_response(self, resp, req): resp.auth.signature = b"signed"
    async def validate_response(self, resp, req): return resp.auth.signature == b"signed"

class DummyStub:
    async def rpc_unary(self, request): return DummyResponse("echo:" + request.msg)

    async def rpc_stream(self, request) -> AsyncIterator[DummyResponse]:
        for i in range(3):
            yield DummyResponse(f"stream:{request.msg}:{i}")

# The test
@pytest.mark.asyncio
async def test_rpc_auth_wrapper_stream_and_unary():
    authorizer = TokenRSAAuthorizerBase()

    # authorizer = DummyAuthorizer()
    raw_stub = DummyStub()

    servicer_stub = AuthRPCWrapper(raw_stub, AuthRole.SERVICER, authorizer)
    client_stub = AuthRPCWrapper(servicer_stub, AuthRole.CLIENT, authorizer)

    # Unary test
    req = DummyRequest("hello")
    resp = await client_stub.rpc_unary(req)
    assert resp is not None
    assert resp.reply == "echo:hello"

    # Streaming test
    stream_req = DummyRequest("world")
    stream = client_stub.rpc_stream(stream_req)
    results = []
    async for r in stream:
        results.append(r.reply)

    assert results == ["stream:world:0", "stream:world:1", "stream:world:2"]
