import * as tf from '@tensorflow/tfjs-core';

import { NetInput, TNetInput, toNetInput } from '../dom';
import { ExtendedFaceFeatureExtractor } from '../faceFeatureExtractor/ExtendedFaceFeatureExtractor';
import { ExtendedFaceFeatureExtractorParams } from '../faceFeatureExtractor/types';
import { FaceProcessor } from '../faceProcessorComplex/FaceProcessor';

export class ExtendedFaceArousalNet extends FaceProcessor<ExtendedFaceFeatureExtractorParams> {

  constructor(faceFeatureExtractor: ExtendedFaceFeatureExtractor = new ExtendedFaceFeatureExtractor()) {
    super('ExtendedFaceArousalNet', faceFeatureExtractor)
  }

  public forwardInput(input: NetInput | tf.Tensor4D): tf.Tensor2D {
    return tf.tidy(() => this.runNet(input))
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input))
  }

  public async predictArousal(input: TNetInput){
    const netInput = await toNetInput(input)
    const out = await this.forwardInput(netInput)
    const probabilitesByBatch = await Promise.all(tf.unstack(out).map(async t => {
      const data = await t.data()
      t.dispose()
      return data
    }))
    out.dispose()

    const predictionsByBatch = probabilitesByBatch // no need to map to FaceExpressions
      //.map(probabilites => new FaceExpressions(probabilites as Float32Array))

    return netInput.isBatchInput
      ? predictionsByBatch
      : predictionsByBatch[0]
  }

  protected getDefaultModelName(): string {
    return 'face_arousal_model_extended'
  }

  protected getClassifierChannelsIn(): number {
    return 512
  }
  protected getClassifierChannelsHidden(): number {
    return 128
  }
  protected getClassifierChannelsOut(): number {
    return 1
  }
}