using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace TensorFlowLite
{
  public class FaceLandmarkDetect : BaseImagePredictor<float>
  {
    public class Result
    {
      public float score;
      public Vector3[] vertices;
    }

    public const int VERTEX_COUNT = 468;

    private float[] output0 = new float[VERTEX_COUNT * 3]; // keypoint
    private float[] output1 = new float[1]; // face flag

    private Result result;
    private Matrix4x4 cropMatrix;
    public Vector2 FaceShift { get; set; } = new Vector2(0, -0.0f);
    //public float FaceScale { get; set; } = 2.8f;
    public float FaceScale { get; set; } = 2.0f;

    // private int inputW = 192;
    // private int inputH = 192;

    public FaceLandmarkDetect(string modelPath) : base(modelPath, true)
    {

      result = new Result()
      {
        score = 0,
        vertices = new Vector3[VERTEX_COUNT],
      };
    }

    public override void Invoke(Texture inputTex)
    {
      throw new System.NotImplementedException("Use Invoke(Texture inputTex, PalmDetect.Palm palm)");
    }

    public void Invoke(Texture inputTex, FaceDetect.Face face)
    {
      var options = resizeOptions;

      cropMatrix = resizer.VertexTransfrom = CalcFaceMatrix(ref face, FaceShift, FaceScale);
      resizer.UVRect = TextureResizer.GetTextureST(inputTex, options);
      RenderTexture rt = resizer.ApplyResize(inputTex, options.width, options.height);
      ToTensor(rt, input0, false);

      interpreter.SetInputTensorData(0, input0);
      interpreter.Invoke();
      interpreter.GetOutputTensorData(0, output0);
      interpreter.GetOutputTensorData(1, output1);

      Interpreter.TensorInfo info = interpreter.GetInputTensorInfo(0);
      //inputW = info.shape.Length[]
      // Debug.Log(info);
      // Debug.Log(options.width);
      // Debug.Log(options.height);

      // Interpreter.TensorInfo info = interpreter.GetOutputTensorInfo(0);
      // Debug.Log(info);
      // Interpreter.TensorInfo info2 = interpreter.GetOutputTensorInfo(1);
      // Debug.Log(info2);
    }

    public Result GetResult()
    {
      // Normalize 0 ~ 255 => 0.0 ~ 1.0
      const float SCALE = 1f / 255f;
      var mtx = cropMatrix.inverse;

      result.score = output1[0];
      for (int i = 0; i < VERTEX_COUNT; i++)
      {
        result.vertices[i] = mtx.MultiplyPoint3x4(new Vector3(
            output0[i * 3],
            output0[i * 3 + 1],
            output0[i * 3 + 2]
        ) * SCALE);
      }
      return result;
    }

    private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
    private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
    private static Matrix4x4 CalcFaceMatrix(ref FaceDetect.Face face, Vector2 shift, float scale)
    {
      // Calc rotation based on 
      // Center of nose - forehead
      const float RAD_90 = 90f * Mathf.PI / 180f;
      var vec = face.keypoints[2] - face.keypoints[3];
      Quaternion rotation = Quaternion.Euler(0, 0, -(RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg);

      // Calc face scale
      float faceScale = Mathf.Max(face.rect.width, face.rect.height) * scale;

      // Calc hand center position
      Vector2 center = face.rect.center + new Vector2(-0.5f, -0.5f);
      center = (Vector2)(rotation * center);
      center += (shift * faceScale);
      center /= faceScale;

      Matrix4x4 trs = Matrix4x4.TRS(
                         new Vector3(-center.x, -center.y, 0),
                         rotation,
                         new Vector3(1 / faceScale, -1 / faceScale, 1)
                      );
      return PUSH_MATRIX * trs * POP_MATRIX;
    }
  }
}