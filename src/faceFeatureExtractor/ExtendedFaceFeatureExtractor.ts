import * as tf from '@tensorflow/tfjs-core';

import { NetInput, TNetInput, toNetInput } from '../dom';
import { NeuralNetwork } from '../NeuralNetwork';
import { normalize } from '../ops';
import { denseBlock4 } from './denseBlock';
import { extractParamsExtended } from './extractParamsExtended'; 
import { extractParamsFromWeigthMapExtended } from './extractParamsFromWeigthMapExtended'; 
import { ExtendedFaceFeatureExtractorParams, IFaceFeatureExtractor } from './types'; 

export class ExtendedFaceFeatureExtractor extends NeuralNetwork<ExtendedFaceFeatureExtractorParams> implements IFaceFeatureExtractor<ExtendedFaceFeatureExtractorParams> {

  constructor() {
    super('ExtendedFaceFeatureExtractor')
  }

  public forwardInput(input: NetInput): tf.Tensor4D {

    const { params } = this

    if (!params) {
      throw new Error('ExtendedFaceFeatureExtractor - load model before inference')
    }

    return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true)
      const meanRgb = [122.782, 117.001, 104.298]
      const normalized = normalize(batchTensor, meanRgb).div(tf.scalar(255)) as tf.Tensor4D

      let out = denseBlock4(normalized, params.dense0, true)
      out = denseBlock4(out, params.dense1)
      out = denseBlock4(out, params.dense2)
      out = denseBlock4(out, params.dense3)
      out = denseBlock4(out, params.dense4, false, false) // an addition block 
      out = tf.avgPool(out, [7, 7], [2, 2], 'valid')

      return out
    })
  }

  public async forward(input: TNetInput): Promise<tf.Tensor4D> {
    return this.forwardInput(await toNetInput(input))
  }

  protected getDefaultModelName(): string {
    return 'face_feature_extractor_extended_model'
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMapExtended(weightMap)
  }

  protected extractParams(weights: Float32Array) {
    return extractParamsExtended(weights)
  }
}