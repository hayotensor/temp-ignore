from dataclasses import dataclass
from typing import List

from substrateinterface import SubstrateInterface

from mesh.substrate.chain_data import SubnetNode
from mesh.substrate.chain_functions import get_subnet_nodes_included, get_subnet_nodes_validator
from mesh.substrate.config import BLOCK_SECS
from mesh.utils import get_logger

logger = get_logger(__name__)

def get_submittable_nodes(substrate: SubstrateInterface, subnet_id: int) -> List:
  result = get_subnet_nodes_validator(
    substrate,
    subnet_id,
  )

  subnet_nodes = SubnetNode.list_from_vec_u8(result["result"])

  return subnet_nodes

def get_included_nodes(substrate: SubstrateInterface, subnet_id: int) -> List:
  result = get_subnet_nodes_included(substrate, subnet_id)

  subnet_nodes_data = SubnetNode.list_from_vec_u8(result["result"])

  return subnet_nodes_data

def get_next_epoch_start_block(
  epochs_length: int,
  block: int
) -> int:
  """Returns next start block for next epoch"""
  return epochs_length + (block - (block % epochs_length))

@dataclass
class EpochProgress:
  current_block: int
  epoch: int
  percent_complete: float
  blocks_remaining: int
  seconds_remaining: int

def get_epoch_progress(current_block: int, epoch_length: int) -> EpochProgress:
  epoch = current_block // epoch_length
  blocks_into_epoch = current_block % epoch_length
  percent_complete = blocks_into_epoch / epoch_length
  blocks_remaining = epoch_length - blocks_into_epoch
  seconds_remaining = blocks_remaining * BLOCK_SECS

  return EpochProgress(
    current_block=current_block,
    epoch=epoch,
    percent_complete=percent_complete,
    blocks_remaining=blocks_remaining,
    seconds_remaining=seconds_remaining
  )
