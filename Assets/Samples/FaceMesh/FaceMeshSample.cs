using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class FaceMeshSample : MonoBehaviour
{
  [SerializeField, FilePopup("*.tflite")] string faceDetectModelFile = "coco_ssd_mobilenet_quant.tflite";
  [SerializeField, FilePopup("*.tflite")] string faceLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";

  [SerializeField] RawImage cameraView = null;
  [SerializeField] Image framePrefab = null;
  [SerializeField] RawImage debugFaceView = null;
  [SerializeField] Mesh jointMesh = null;
  [SerializeField] Material jointMaterial = null;

  WebCamTexture webcamTexture;
  FaceDetect faceDetect;
  FaceLandmarkDetect landmarkDetect;

  Image[] frames;
  Vector3[] rtCorners = new Vector3[4]; // just cache for GetWorldCorners
  Matrix4x4[] vertexMatrices = new Matrix4x4[FaceLandmarkDetect.VERTEX_COUNT];


  [SerializeField] Texture2D testTexture = null;

  bool doDetectNextFrame = true;
  bool faceFound = false;

  FaceDetect.Face lastFace;

  // Start is called before the first frame update
  void Start()
  {
    string faceDetectPath = Path.Combine(Application.streamingAssetsPath, faceDetectModelFile);
    faceDetect = new FaceDetect(faceDetectPath);

    string landmarkPath = Path.Combine(Application.streamingAssetsPath, faceLandmarkModelFile);
    landmarkDetect = new FaceLandmarkDetect(landmarkPath);
    //Debug.Log($"landmark dimension: {landmarkDetect.Dim}");

    string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
    {
      isFrontFacing = false,
      kind = WebCamKind.WideAngle,
    });
    webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
    // webcamTexture = new WebCamTexture(cameraName, 640, 480, 30);
    cameraView.texture = webcamTexture;
    webcamTexture.Play();
    Debug.Log($"Starting camera: {cameraName}");

    // Init frames
    frames = new Image[FaceDetect.MAX_FACE_NUM];
    var parent = cameraView.transform;
    for (int i = 0; i < frames.Length; i++)
    {
      frames[i] = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, parent);
      frames[i].transform.localPosition = Vector3.zero;
    }
  }
  void OnDestroy()
  {
    webcamTexture?.Stop();
    faceDetect?.Dispose();
    landmarkDetect?.Dispose();
  }

  // Update is called once per frame
  void Update()
  {
    if (Input.GetKeyDown("space"))
    {
      doDetectNextFrame = true;
    }

    var resizeOptions = faceDetect.ResizeOptions;
    resizeOptions.rotationDegree = webcamTexture.videoRotationAngle;
    faceDetect.ResizeOptions = resizeOptions;

    if (doDetectNextFrame)
    {
      faceDetect.Invoke(webcamTexture);
      cameraView.material = faceDetect.transformMat;

      var faces = faceDetect.GetResults(0.75f, 0.3f);
      UpdateFrame(faces);

      doDetectNextFrame = false;

      if (faces.Count <= 0)
      {
        faceFound = false;
      }
      else
      {
        faceFound = true;
        lastFace = faces[0];
      }
    }


    if (faceFound)
    {
      // Detect only first face
      landmarkDetect.Invoke(webcamTexture, lastFace);
      debugFaceView.texture = landmarkDetect.inputTex;

      var vertices = landmarkDetect.GetResult().vertices;
      DrawVertices(vertices);
    }

    /*
        landmarkDetect.Invoke(testTexture);
        debugFaceView.texture = landmarkDetect.inputTex;
        cameraView.texture = landmarkDetect.inputTex;

        var vertices = landmarkDetect.GetResult().vertices;
        DrawVertices(vertices);
        */

  }
  void UpdateFrame(List<FaceDetect.Face> faces)
  {
    var size = ((RectTransform)cameraView.transform).rect.size;
    for (int i = 0; i < faces.Count; i++)
    {
      frames[i].gameObject.SetActive(true);
      SetFrame(frames[i], faces[i], size);
    }
    for (int i = faces.Count; i < frames.Length; i++)
    {
      frames[i].gameObject.SetActive(false);
    }
  }
  void SetFrame(Graphic frame, FaceDetect.Face face, Vector2 size)
  {
    var rt = frame.transform as RectTransform;
    var p = face.rect.position;
    p.y = 1.0f - p.y; // invert Y
    rt.anchoredPosition = p * size - size * 0.5f;
    rt.sizeDelta = face.rect.size * size;

    var kpOffset = -rt.anchoredPosition + new Vector2(-rt.sizeDelta.x, rt.sizeDelta.y) * 0.5f;
    for (int i = 0; i < 6; i++)
    {
      var child = (RectTransform)rt.GetChild(i);
      var kp = face.keypoints[i];
      kp.y = 1.0f - kp.y; // invert Y
      child.anchoredPosition = (kp * size - size * 0.5f) + kpOffset;
    }
  }

  void DrawVertices(Vector3[] vertices)
  {
    var rt = cameraView.transform as RectTransform;
    rt.GetWorldCorners(rtCorners);
    Vector3 min = rtCorners[0];
    Vector3 max = rtCorners[2];
    float zScale = max.x - min.x;

    var rotation = Quaternion.identity;
    var scale = Vector3.one * 0.1f;
    for (int i = 0; i < FaceLandmarkDetect.VERTEX_COUNT; i++)
    {
      var p = vertices[i];

      p = MathTF.Leap3(min, max, p);
      p.z += (vertices[i].z - 0.5f) * zScale;
      var mtx = Matrix4x4.TRS(p, rotation, scale);
      vertexMatrices[i] = mtx;
    }
    Graphics.DrawMeshInstanced(jointMesh, 0, jointMaterial, vertexMatrices);
  }

}
