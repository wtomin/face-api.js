import * as tf from '@tensorflow/tfjs-core';

import { fullyConnectedLayer } from '../common/fullyConnectedLayer';
import { NetInput } from '../dom';
import {
  FaceFeatureExtractorParams,
  IFaceFeatureExtractor,
  TinyFaceFeatureExtractorParams,
  ExtendedFaceFeatureExtractorParams,
} from '../faceFeatureExtractor/types';
import { NeuralNetwork } from '../NeuralNetwork';
import { extractParams } from './extractParams';
import { extractParamsFromWeigthMap } from './extractParamsFromWeigthMap';
import { NetParams } from './types';
import { seperateWeightMaps } from './util';

export abstract class FaceProcessor<
  TExtractorParams extends FaceFeatureExtractorParams | TinyFaceFeatureExtractorParams | ExtendedFaceFeatureExtractorParams
>
  extends NeuralNetwork<NetParams> {

  protected _faceFeatureExtractor: IFaceFeatureExtractor<TExtractorParams>

  constructor(_name: string, faceFeatureExtractor: IFaceFeatureExtractor<TExtractorParams>) {
    super(_name)
    this._faceFeatureExtractor = faceFeatureExtractor
  }

  public get faceFeatureExtractor(): IFaceFeatureExtractor<TExtractorParams> {
    return this._faceFeatureExtractor
  }

  protected abstract getDefaultModelName(): string
  protected abstract getClassifierChannelsIn(): number
  protected abstract getClassifierChannelsHidden(): number
  protected abstract getClassifierChannelsOut(): number

  public runNet(input: NetInput | tf.Tensor4D): tf.Tensor2D {

    const { params } = this

    if (!params) {
      throw new Error(`${this._name} - load model before inference`)
    }

    return tf.tidy(() => {
      const bottleneckFeatures = input instanceof NetInput
        ? this.faceFeatureExtractor.forwardInput(input)
        : input
      return fullyConnectedLayer(tf.relu(fullyConnectedLayer(bottleneckFeatures.as2D(bottleneckFeatures.shape[0], -1), params.fc.fc_1)),params.fc.fc_2) 
      
    })
  }

  public dispose(throwOnRedispose: boolean = true) {
    this.faceFeatureExtractor.dispose(throwOnRedispose)
    super.dispose(throwOnRedispose)
  }

  public loadClassifierParams(weights: Float32Array) {
    const { params, paramMappings } = this.extractClassifierParams(weights)
    this._params = params
    this._paramMappings = paramMappings
  }

  public extractClassifierParams(weights: Float32Array) {
    return extractParams(weights, this.getClassifierChannelsIn(), this.getClassifierChannelsHidden(), this.getClassifierChannelsOut())
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {

    const { featureExtractorMap, classifierMap } = seperateWeightMaps(weightMap)

    this.faceFeatureExtractor.loadFromWeightMap(featureExtractorMap)

    return extractParamsFromWeigthMap(classifierMap)
  }

  protected extractParams(weights: Float32Array) {

    const cIn = this.getClassifierChannelsIn()
    const cHidden = this.getClassifierChannelsHidden()
    const cOut = this.getClassifierChannelsOut()
    const classifierWeightSize = (cIn * cHidden ) + cHidden +  (cHidden * cOut) +cOut 

    const featureExtractorWeights = weights.slice(0, weights.length - classifierWeightSize)
    const classifierWeights = weights.slice(weights.length - classifierWeightSize)

    this.faceFeatureExtractor.extractWeights(featureExtractorWeights)
    return this.extractClassifierParams(classifierWeights)
  }
}