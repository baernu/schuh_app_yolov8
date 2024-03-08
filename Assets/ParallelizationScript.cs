using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UI = UnityEngine.UI;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Linq;

public class ParallelizationScript : MonoBehaviour
{

    public NNModel _model;
    public Texture2D _image;
    public UI.RawImage _imageView;

    [SerializeField] Vector2 webCamResolution = new Vector2(1920, 1080);
    [SerializeField] string webCamName;
    [SerializeField] Material material;
    WebCamTexture webCamTexture;
    RenderTexture inputRT;
    Tensor output0;
    Tensor inputTensor;
    IWorker worker;
    int count = 0;

    private int _resizeLength = 640;


    private List<Color> _colorList = new List<Color>() {
        Color.red,     // 0: big toe left
        Color.cyan,    // 1: big toe right
        Color.blue,    // 2: little toe left
        Color.magenta,    // 3: little toe right
        Color.white,    // 4: heel left
        Color.green,   // 5: heel right
        new Color(0.5f,0.5f,0.5f,1), // 6: inside ankle left
        new Color(0.5f,0,0.5f,1),   // 7: inside ankle right
        new Color(0,0.5f,0.5f,1), // 8: outside ankle left
        new Color(0,0.5f,0,1),   // 9: outside ankle right
        new Color(0.25f,0.25f,0.5f,1), // 10: ankle kink left
        new Color(0.75f,0.25f,0,1)    // 11: ankle kink right
       
    };


    // Start is called before the first frame update
    void Start()
    {

        Application.targetFrameRate = 70;
        // onnx
        var model = ModelLoader.Load(_model);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        webCamTexture = new WebCamTexture(webCamName, (int)webCamResolution.x, (int)webCamResolution.y);
        webCamTexture.Play();
        inputRT = new RenderTexture((int)webCamResolution.x, (int)webCamResolution.y, 0);

    }

    void LateUpdate()
    {

        if (!webCamTexture.didUpdateThisFrame) return;

        if (count == 8)
        {
            var aspect1 = (float)webCamTexture.width / webCamTexture.height;
            var aspect2 = (float)inputRT.width / inputRT.height;
            var aspectGap = aspect2 / aspect1;

            var vMirrored = webCamTexture.videoVerticallyMirrored;
            var scale = new Vector2(aspectGap, vMirrored ? -1 : 1);
            var offset = new Vector2((1 - aspectGap) / 2, vMirrored ? 1 : 0);


            Graphics.Blit(webCamTexture, inputRT, scale, offset);
            //material.mainTexture = inputRT;

            _image = toTexture2D(inputRT);


            var texture = ResizedTexture(_image, _resizeLength, _resizeLength);
            inputTensor = new Tensor(texture, channels: 3);


            worker.Execute(inputTensor);


            output0 = worker.PeekOutput("output0");
            List<DetectionResult> ditects = ParseOutputs(output0, 0.5f, 0.5f);
            

            inputTensor.Dispose();
            output0.Dispose();



            float scaleX = _image.width / (float)_resizeLength;
            float scaleY = _image.height / (float)_resizeLength;

            var image = ResizedTexture(_image, _image.width, _image.height);
            foreach (DetectionResult ditect in ditects)
            {

                int x1 = (int)(ditect.x1 * scaleX);
                int x2 = (int)(ditect.x2 * scaleX);
                int y1 = (int)(ditect.y1 * scaleY);
                int y2 = (int)(ditect.y2 * scaleY);

                for (int x = x1; x < x2; x++)
                {
                    image.SetPixel(x, _image.height - y1, Color.red);
                    image.SetPixel(x, _image.height - (y1 - 1), Color.red);
                    image.SetPixel(x, _image.height - y2, Color.red);
                    image.SetPixel(x, _image.height - (y2 + 1), Color.red);
                }
                for (int y = y1; y < y2; y++)
                {
                    image.SetPixel(x1, _image.height - y, Color.red);
                    image.SetPixel(x1 - 1, _image.height - y, Color.red);
                    image.SetPixel(x2, _image.height - y, Color.red);
                    image.SetPixel(x2 + 1, _image.height - y, Color.red);
                }



                int point = 0;
                foreach (KeyPoint kp in ditect.keypoints)
                {

                    int centerX = (int)(kp.x * scaleX);
                    int centerY = (int)(kp.y * scaleY);
                    for (int x = centerX - 3; x < centerX + 3; x++)
                    {
                        for (int y = centerY - 3; y < centerY + 3; y++)
                        {

                            image.SetPixel(x, _image.height - y, _colorList[point]);
                        }
                    }
                    point++;
                }
            }
            image.Apply();

            //_imageView.texture = image;
            material.mainTexture = image;
            count = 0;
            return;
        }
        material.mainTexture = webCamTexture;

        count++;
    }

    void OnDestroy()
    {

        if (webCamTexture != null) Destroy(webCamTexture);
        if (inputRT != null) Destroy(inputRT);
        worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();
    }




    private List<DetectionResult> ParseOutputs(Tensor output0, float threshold, float iouThres)
    {

        int outputWidth = output0.shape.width;

        List<DetectionResult> candidateDitects = new List<DetectionResult>();

        List<DetectionResult> ditects = new List<DetectionResult>();

        for (int i = 0; i < outputWidth; i++)
        {

            var result = new DetectionResult(output0, i);

            if (result.score < threshold)
            {
                continue;
            }

            candidateDitects.Add(result);
        }

        // NonMaxSuppression
        while (candidateDitects.Count > 0)
        {
            int idx = 0;
            float maxScore = 0.0f;
            for (int i = 0; i < candidateDitects.Count; i++)
            {
                if (candidateDitects[i].score > maxScore)
                {
                    idx = i;
                    maxScore = candidateDitects[i].score;
                }
            }

            // score
            var cand = candidateDitects[idx];
            candidateDitects.RemoveAt(idx);


            ditects.Add(cand);

            List<int> deletes = new List<int>();
            for (int i = 0; i < candidateDitects.Count; i++)
            {
                // IOU
                float iou = Iou(cand, candidateDitects[i]);
                if (iou >= iouThres)
                {
                    deletes.Add(i);
                }
            }
            for (int i = deletes.Count - 1; i >= 0; i--)
            {
                candidateDitects.RemoveAt(deletes[i]);
            }

        }

        return ditects;

    }
    

    private float Iou(DetectionResult boxA, DetectionResult boxB)
    {
        if ((boxA.x1 == boxB.x1) && (boxA.x2 == boxB.x2) && (boxA.y1 == boxB.y1) && (boxA.y2 == boxB.y2))
        {
            return 1.0f;

        }
        else if (((boxA.x1 <= boxB.x1 && boxA.x2 > boxB.x1) || (boxA.x1 >= boxB.x1 && boxB.x2 > boxA.x1))
            && ((boxA.y1 <= boxB.y1 && boxA.y2 > boxB.y1) || (boxA.y1 >= boxB.y1 && boxB.y2 > boxA.y1)))
        {
            float intersection = (Mathf.Min(boxA.x2, boxB.x2) - Mathf.Max(boxA.x1, boxB.x1))
                * (Mathf.Min(boxA.y2, boxB.y2) - Mathf.Max(boxA.y1, boxB.y1));
            float union = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1) + (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1) - intersection;
            return (intersection / union);
        }

        return 0.0f;
    }



    private static Texture2D ResizedTexture(Texture2D texture, int width, int height)
    {

        var rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(texture, rt);

        var preRt = RenderTexture.active;
        RenderTexture.active = rt;
        var resizedTexture = new Texture2D(width, height);
        resizedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        resizedTexture.Apply();
        RenderTexture.active = preRt;
        RenderTexture.ReleaseTemporary(rt);
        return resizedTexture;
    }

    Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
        // ReadPixels looks at the active RenderTexture.
        RenderTexture.active = rTex;
        tex.ReadPixels(new UnityEngine.Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }

}


class DetectionResult
{
    public float x1 { get; }
    public float y1 { get; }
    public float x2 { get; }
    public float y2 { get; }
    public float score { get; }
    public List<KeyPoint> keypoints { get; }

    public DetectionResult(Tensor t, int idx)
    {

        float halfWidth = t[0, 0, idx, 2] / 2;
        float halfHeight = t[0, 0, idx, 3] / 2;
        x1 = t[0, 0, idx, 0] - halfWidth;
        y1 = t[0, 0, idx, 1] - halfHeight;
        x2 = t[0, 0, idx, 0] + halfWidth;
        y2 = t[0, 0, idx, 1] + halfHeight;
        score = t[0, 0, idx, 4];


        int channels = t.shape.channels;
        keypoints = new List<KeyPoint>();
        for (int point = 5; point < channels; point += 3)
        {
            keypoints.Add(new KeyPoint(
                t[0, 0, idx, point],
                t[0, 0, idx, point + 1],
                t[0, 0, idx, point + 2]
            ));
        }
    }

}


class KeyPoint
{
    public float x { get; }
    public float y { get; }
    public float visible { get; }

    public KeyPoint(float x, float y, float visible)
    {
        this.x = x;
        this.y = y;
        this.visible = visible;
    }

}