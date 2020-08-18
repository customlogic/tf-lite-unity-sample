using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


namespace TensorFlowLite
{

  public class FaceDetect : BaseImagePredictor<float>
  {
    private Vector2[] anchors;

    public enum FaceKeyID
    {
      kRightEye = 0,  //  0
      kLeftEye,       //  1
      kNose,          //  2
      kMouth,         //  3
      kRightEar,      //  4
      kLeftEar,       //  5
      kFaceKeyNum
    }

    public struct Face
    {
      public float score;
      public Vector2 topleft;
      public Vector2 bottomRight;
      public Vector2[] keys;

      public float rotation;
      public Rect rect;
      public Vector2[] keypoints;
    }

    public const int MAX_FACE_NUM = 4;


    // classificators / scores
    private float[] output0 = new float[896];

    // regressors / points
    private float[,] output1 = new float[896, 16];

    private List<Face> results = new List<Face>();

    private static Vector2[] generateAnchors(int inputW, int inputH)
    {
      /* ANCHORS_CONFIG  */
      int[] strides = new int[] { 8, 16 };
      int[] anchors = new int[] { 2, 6 };

      int numtotal = 0;

      var anchorList = new List<Vector2>();
      for (int i = 0; i < 2; i++)
      {
        int stride = strides[i];
        int gridCols = (inputW + stride - 1) / stride;
        int gridRows = (inputH + stride - 1) / stride;
        int anchorNum = anchors[i];

        Vector2 anchor;
        for (int gridY = 0; gridY < gridRows; gridY++)
        {
          anchor.y = stride * (gridY + 0.5f);
          for (int gridX = 0; gridX < gridCols; gridX++)
          {
            anchor.x = stride * (gridX + 0.5f);
            for (int n = 0; n < anchorNum; n++)
            {
              anchorList.Add(anchor);
              numtotal++;
            }
          }
        }
      }
      return anchorList.ToArray();
    }

    public FaceDetect(string modelPath) : base(modelPath, true)
    {
      int inputW = 128;
      int inputH = 128;
      anchors = generateAnchors(inputW, inputH);

      Debug.Log(anchors.Length);
    }


    public override void Invoke(Texture inputTex)
    {
      ToTensor(inputTex, input0);

      interpreter.SetInputTensorData(0, input0);
      interpreter.Invoke();

      // Interpreter.TensorInfo info = interpreter.GetOutputTensorInfo(0);
      // Debug.Log(info);
      // Interpreter.TensorInfo info2 = interpreter.GetOutputTensorInfo(1);
      // Debug.Log(info2);
      interpreter.GetOutputTensorData(0, output0);
      interpreter.GetOutputTensorData(1, output1);
    }

    public List<Face> GetResults(float scoreThreshold = 0.75f, float iouThreshold = 0.3f)
    {
      results.Clear();


      for (int i = 0; i < anchors.Length; i++)
      {
        float score = MathTF.Sigmoid(output0[i]);
        if (score < scoreThreshold)
        {
          continue;
        }

        Vector2 anchor = anchors[i];

        float sx = output1[i, 0];
        float sy = output1[i, 1];
        float w = output1[i, 2];
        float h = output1[i, 3];

        float cx = sx + anchor.x;
        float cy = sy + anchor.y;

        cx /= (float)width;
        cy /= (float)height;
        w /= (float)width;
        h /= (float)height;

        Vector2 topleft = new Vector2();
        Vector2 btmright = new Vector2();
        topleft.x = cx - w * 0.5f;
        topleft.y = cy - h * 0.5f;
        btmright.x = cx + w * 0.5f;
        btmright.y = cy + h * 0.5f;

        Face face = new Face();
        face.score = score;
        face.topleft = topleft;
        face.bottomRight = btmright;

        var keypoints = new Vector2[6];
        for (int j = 0; j < 6; j++)
        {
          float lx = output1[i, 4 + (2 * j) + 0];
          float ly = output1[i, 4 + (2 * j) + 1];
          lx += anchor.x;
          ly += anchor.y;
          lx /= (float)width;
          ly /= (float)height;
          keypoints[j] = new Vector2(lx, ly);
        }

        face.rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h);
        face.keypoints = keypoints;

        results.Add(face);
      }

      return NonMaxSuppression(results, iouThreshold);
    }

    private static List<Face> NonMaxSuppression(List<Face> faces, float iou_threshold)
    {
      var filtered = new List<Face>();

      foreach (Face originalFace in faces.OrderByDescending(o => o.score))
      {
        bool ignore_candidate = false;
        foreach (Face newFace in filtered)
        {
          float iou = CalcIntersectionOverUnion(originalFace.rect, newFace.rect);
          if (iou >= iou_threshold)
          {
            ignore_candidate = true;
            break;
          }
        }

        if (!ignore_candidate)
        {
          filtered.Add(originalFace);
          if (filtered.Count >= MAX_FACE_NUM)
          {
            break;
          }
        }
      }

      return filtered;
    }

    private static float CalcIntersectionOverUnion(Rect rect0, Rect rect1)
    {
      float sx0 = rect0.xMin;
      float sy0 = rect0.yMin;
      float ex0 = rect0.xMax;
      float ey0 = rect0.yMax;
      float sx1 = rect1.xMin;
      float sy1 = rect1.yMin;
      float ex1 = rect1.xMax;
      float ey1 = rect1.yMax;

      float xmin0 = Mathf.Min(sx0, ex0);
      float ymin0 = Mathf.Min(sy0, ey0);
      float xmax0 = Mathf.Max(sx0, ex0);
      float ymax0 = Mathf.Max(sy0, ey0);
      float xmin1 = Mathf.Min(sx1, ex1);
      float ymin1 = Mathf.Min(sy1, ey1);
      float xmax1 = Mathf.Max(sx1, ex1);
      float ymax1 = Mathf.Max(sy1, ey1);

      float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
      float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
      if (area0 <= 0 || area1 <= 0)
      {
        return 0.0f;
      }

      float intersect_xmin = Mathf.Max(xmin0, xmin1);
      float intersect_ymin = Mathf.Max(ymin0, ymin1);
      float intersect_xmax = Mathf.Min(xmax0, xmax1);
      float intersect_ymax = Mathf.Min(ymax0, ymax1);

      float intersect_area = Mathf.Max(intersect_ymax - intersect_ymin, 0.0f) *
                             Mathf.Max(intersect_xmax - intersect_xmin, 0.0f);

      return intersect_area / (area0 + area1 - intersect_area);
    }
  }
}