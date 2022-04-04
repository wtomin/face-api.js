import * as tf from '@tensorflow/tfjs-core';

import { disposeUnusedWeightTensors, ParamMapping } from '../common';
import { loadParamsFactory } from './loadParamsFactory';
import { ExtendedFaceFeatureExtractorParams } from './types';

export function extractParamsFromWeigthMapExtended(
  weightMap: tf.NamedTensorMap
): { params: ExtendedFaceFeatureExtractorParams, paramMappings: ParamMapping[] } {

  const paramMappings: ParamMapping[] = []

  const {
    extractDenseBlock4Params
  } = loadParamsFactory(weightMap, paramMappings)

  const params = {
    dense0: extractDenseBlock4Params('dense0', true),
    dense1: extractDenseBlock4Params('dense1'),
    dense2: extractDenseBlock4Params('dense2'),
    dense3: extractDenseBlock4Params('dense3'),
    dense4: extractDenseBlock4Params('dense4', false, false) // isfirstlayer=false, isScaleDown=False
  }

  disposeUnusedWeightTensors(weightMap, paramMappings)

  return { params, paramMappings }
}