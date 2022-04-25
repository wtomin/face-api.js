import { extractFCParamsFactory, extractWeightsFactory, ParamMapping } from '../common';
import { NetParams } from './types';

export function extractParams(weights: Float32Array, channelsIn: number, channelsHidden: number, channelsOut: number): { params: NetParams, paramMappings: ParamMapping[] } {

  const paramMappings: ParamMapping[] = []

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  const extractFCParams = extractFCParamsFactory(extractWeights, paramMappings)

  const fc_1 = extractFCParams(channelsIn, channelsHidden, 'fc_1')

  const fc_2 = extractFCParams(channelsHidden, channelsOut, 'fc_2')

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  return {
    paramMappings,
    params: { fc: { fc_1, fc_2 } }
  }
}