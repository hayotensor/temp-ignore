## Container: A decentralized container for Hypertensor.

This Hypertensor container is an AI application built for inference in a decentralized environment. This is the OPEN OpenAI.

##### Roles
- Hoster: Hosts the model, performs inference for clients, validates other hosters and validators.
- Validator: Validates hoster nodes, and other validators.

##### Specs
- Proof-of-stake.
- Ed25519 signature authentication between RPC methods.
- Built for blazing-fast and decentralized inference.

##### Consensus
Both hosters and validators use a commit-reveal schema for consensus. Consensus takes place directly in the subnet itself utilizing the decentralized record storage, and the scores are submitted to the blockchain.

On each epoch, the chosen validator (chosen by Hypertensor blockchain nodes) uploads their scores from the previous epoch to the blockchain and submits a randomized tensor (prompt) to the DHT. If no tensor is submitted by a certain point in the epoch, any node can take over this task.

Commit-reveal is used to ensure no nodes are able to copy scores from each other. Due to randomized tensors being used, scores can differ from epoch to epoch slightly, making copying the previous epochs data also a challenge.
 
![alt text](https://github.com/hayotensor/temp-ignore/commit-reveal-inference-subnet.png "Logo Title Text 1")

##### Hoster

- Hosters get the random tensor from the DHT and run inference.
- A commit of the output is stored in the DHT as a hash during the commit phase.
- In the reveal phase, they store the salt in the DHT that was used to commit.
- Hosters then get all hosters' commits and reveals, including themselves, to unhash the commits to verify and score each node (See explanation below in Validator).

##### Validator
- Validators then get all hosters' commits and reveals to unhash the commits.
- Validators use this data to score each hoster by computing the accuracy for each hoster based on proximity to the mean output. The node does this by comparing each successful hoster's output tensor to the mean tensor of all valid outputs using the L2 norm. Scores are inversely proportional to the distance from the mean — the closer a hoster's output is to the average, the higher the score.
- Each validator commits the scores to the DHT as a hash in the current epoch.
- In the following epoch, they store the salt of the commit to the DHT.
  - These reveals are used to score the validators in the next epoch (validators are rewarded and scored after their first epoch due to reveals being done on the following epoch to the commit).

The elected validator for the epoch then retrieves the hoster scores they stored, generates validator scores based on their commit and reveal accuracy based on the proximity to the mean output. They then submit these scores to Hypertensor. Each other non-elected node is an attestor that generates the same scores, and then optionally attests based on the accuracy of the elected validators submission versus their own.

Since every node has access to the same data, the scores are deterministic and therefor all nodes should have identical data. Therefore, there are no discrepancies.

---

## Installation From source

### From source

To install this container from source, simply run the following:

```
git clone https://github.com/hypertensor-blockchain/container.git
cd container
python -m venv .venv
source .venv/bin/activate
pip install .
```

If you would like to verify that your installation is working properly, you can install with `pip install .[dev]`
instead. Then, you can run the tests with `pytest tests/`.

By default, the contatiner uses the precompiled binary of
the [go-libp2p-daemon](https://github.com/learning-at-home/go-libp2p-daemon) library. If you face compatibility issues
or want to build the binary yourself, you can recompile it by running `pip install . --global-option="--buildgo"`.
Before running the compilation, please ensure that your machine has a recent version
of [Go toolchain](https://golang.org/doc/install) (1.15 or 1.16 are supported).

### System requirements

- __Linux__ is the default OS for which the container is developed and tested. We recommend Ubuntu 18.04+ (64-bit), but
  other 64-bit distros should work as well. Legacy 32-bit is not recommended.
- __macOS__ is partially supported.
  If you have issues, you can run the container using [Docker](https://docs.docker.com/desktop/mac/install/) instead.
  We recommend using [our Docker image](https://hub.docker.com/r/hypertensor-blockchain/mesh).
- __Windows 10+ (experimental)__ can run the container
  using [WSL](https://docs.microsoft.com/ru-ru/windows/wsl/install-win10). You can configure WSL to use GPU by
  following sections 1–3 of [this guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) by NVIDIA. After
  that, you can simply follow the instructions above to install with pip or from source.

---

## Documentation

### Preliminaries

##### Generate private keys

  - This will create 3 private key files for your peer
      - `peer_id`: Main peer ID for communication and signing
      - `bootstrap_peer_id`: (Optional usage) Peer ID to be used as a bootstrap node.
      - `client_peer_id`: (Optional usage) Peer ID to be used as a client. This is for those who want to build frontends to interact with the subnet.

##### Register & Stake on Hypertensor
  - Call `register_subnet_node`
  - Retrieve your `start_epoch` by querying your SubnetNodesData storage element on polkadot.js with your subnet node ID. This is the epoch you must activate your node on + the grace period

### Run node 
(See below for role options)

##### Activate node
  - Call `activate_subnet_node` in Hypertensor on your start epoch up to the grace period.

### Hoster Role
##### Start Node
```bash
mesh-test TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host_maddrs /ip4/0.0.0.0/tcp/31330 /ip4/0.0.0.0/udp/31330/quic --announce_maddrs /ip4/{IP}/tcp/{PORT} /ip4/{IP}/udp/{PORT}/quic --new_swarm --hoster --identity_path {PRIVATE_KEY_PATH} --subnet_id {SUBNET_ID} --subnet_node_id {SUBNET_NODE_ID}
```

### Validator Role
##### Start Node
```bash
mesh-test TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host_maddrs /ip4/0.0.0.0/tcp/31330 /ip4/0.0.0.0/udp/31330/quic --announce_maddrs /ip4/{IP}/tcp/{PORT} /ip4/{IP}/udp/{PORT}/quic --new_swarm --hoster --identity_path {PRIVATE_KEY_PATH}
```

---

## Future

- Migrate to py-libp2p over daemon once productionized.
- Transition to or integrate EZKL	(https://github.com/zkonduit/ezkl) for hoster validation.

---

## Contributing

The container is currently at the active development stage, and we welcome all contributions. Everything, from bug fixes and documentation improvements to entirely new features, is appreciated.

If you want to contribute to this container but don't know where to start, take a look at the unresolved [issues](https://github.com/hypertensor-blockchain/container/issues). 

Open a new issue or join [our chat room](https://discord.gg/bY7NUEweQp) in case you want to discuss new functionality or report a possible bug. Bug fixes are always welcome, but new features should be preferably discussed with maintainers beforehand.