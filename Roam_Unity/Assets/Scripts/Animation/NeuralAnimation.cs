using UnityEngine;
using DeepLearning;
using System.Collections;
#if UNITY_EDITOR
using UnityEditor;
#endif

public abstract class NeuralAnimation : MonoBehaviour
{

    public enum FPS { Thirty, Sixty, Fifty, TwentyFive, Fifteen }
    public NeuralNetwork NeuralNetwork = null;
    public Actor Actor;

    public float AnimationTime { get; private set; }
    public float PostprocessingTime { get; private set; }
    public FPS Framerate = FPS.TwentyFive;

    protected abstract void Setup();

    protected abstract void Feed();
    protected abstract void Read();
    protected abstract void Postprocess();
    protected abstract void OnGUIDerived();
    protected abstract void OnRenderObjectDerived();


    void Start()
    {
        Setup();

    }


    IEnumerator PauseBeforeStart()
    {
       yield return new WaitForSeconds(4f);
       Debug.Log("Finished Waiting");
    }

    void LateUpdate()
    {
        Utility.SetFPS(Mathf.RoundToInt(GetFramerate()));

        if (NeuralNetwork != null && NeuralNetwork.Setup)
        {
            System.DateTime t1 = Utility.GetTimestamp();
      
            NeuralNetwork.ResetPivot();
    
            Feed();
            NeuralNetwork.Predict();
            NeuralNetwork.ResetPivot();
            Read();

            AnimationTime = (float)Utility.GetElapsedTime(t1);

            System.DateTime t2 = Utility.GetTimestamp();
            Postprocess();
            PostprocessingTime = (float)Utility.GetElapsedTime(t2);
        } 
    }

    void OnGUI()
    {

        OnGUIDerived();
   
    }

    void OnRenderObject()
    {

        if (Application.isPlaying)
        {
            OnRenderObjectDerived();
        }
   
    }

    public float GetFramerate()
    {
        switch (Framerate)
        {
            case FPS.Thirty:
                return 30f;
            case FPS.Sixty:
                return 60f;
            case FPS.Fifty:
                return 50f;
            case FPS.TwentyFive:
                return 25f;
            case FPS.Fifteen:
                return 15f;
        }
        return 1f;
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(NeuralAnimation), true)]
    public class NeuralAnimation_Editor : Editor
    {

        public NeuralAnimation Target;

        void Awake()
        {
            Target = (NeuralAnimation)target;
        }

        public override void OnInspectorGUI()
        {
            Undo.RecordObject(Target, Target.name);

            DrawDefaultInspector();

            EditorGUILayout.HelpBox("Animation: " + 1000f * Target.AnimationTime + "ms", MessageType.None);
            EditorGUILayout.HelpBox("Postprocessing: " + 1000f * Target.PostprocessingTime + "ms", MessageType.None);

            if (GUI.changed)
            {
                EditorUtility.SetDirty(Target);
            }
        }

    }
#endif

}
