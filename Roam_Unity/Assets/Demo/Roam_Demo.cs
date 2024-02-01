using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;



public class Roam_Demo : NeuralAnimation
{
 
    private Controller Controller;
    private float[] Signals = new float[0];
    private UltimateIK.Model RightFootIK, LeftFootIK, RightHandIK, LeftHandIK, SpineIK;
    private float UserControl = 0f;
    private float NetworkControl = 0f;
    private TimeSeries TimeSeriesSparse; 
    private TimeSeries.Root RootSeries;
    private TimeSeries.Style StyleSeries;
    private TimeSeries.Goal GoalSeries;
    private TimeSeries.Contact ContactSeries;
    private TimeSeries.Phase PhaseSeries;
    private Vector3[] positions; 
    private Matrix4x4[] RootPrediction;
    private Matrix4x4[] GoalPrediction;
    private bool IsInteracting = false;
    private bool IdleComplete = false;
    private int IdleCount = 0;
    private bool WalkComplete = false;
    private bool SitComplete = false;
    private bool LieComplete = false;
    private bool ActivateGoal = false;
    public bool ShowBiDirectional = true;
    public bool ShowRoot = true;
    public bool ShowGoal = true;
    public bool ShowCurrent = true;
    public bool ShowPhase = false;
    public bool ShowContacts = false;
    public bool ShowGUI = true;
    public bool ShowMoevement = true;
    public bool HighLevel = false;
    private bool InverseBlend = true;
    public float StartRadius = 3f;
    public float StartAngle = 0f;
    public Actor DisplayActor;
    public Actor FinalActor;
    public Actor InverseActor;
    private float FootDistanceThreshold = 0.3f;
    private string PoseInitPath = "";
    private string GoalPoseAction = "Sit";
    public GameObject BatchEvalCollection;
    public int BatchEvalObjTotalCount = 0;
    public int BatchEvalObjCurrentCount = 0;
    private bool SavePositionError = false;
    public int TotalFrames = 300;
    private int FrameCount = 0;
    public string ExportPath = "ErrorSave/";

    float[] hand_list = new float[0];
    float[] all_joint_list = new float[0];
    float[] hip_list = new float[0];

    private Texture Forward, Left, Right, Back, TurnLeft, TurnRight, Disc, SitUI, LieUI;


    Matrix4x4 GetGoalRoot(bool project=true)
    {
        Actor currActor = FinalActor;
        if (!HighLevel){
            currActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        }
        
        // Update Root rotation
        Vector3 v1 = Vector3.ProjectOnPlane(currActor.GetBoneTransformation("RightUpLeg").GetPosition() - currActor.GetBoneTransformation("LeftUpLeg").GetPosition(), Vector3.up).normalized;
        Vector3 v2 = Vector3.ProjectOnPlane(currActor.GetBoneTransformation("RightShoulder").GetPosition() - currActor.GetBoneTransformation("LeftShoulder").GetPosition(), Vector3.up).normalized;
        Vector3 v = (v1 + v2).normalized;
        Vector3 forward = -Vector3.ProjectOnPlane(Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
        Quaternion root_rotation = forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);

        Vector3 root_position = currActor.GetBoneTransformation("Hips").GetPosition();
        if (project){
            LayerMask mask = LayerMask.GetMask("Ground");
            root_position = Utility.ProjectGround(currActor.GetBoneTransformation("Hips").GetPosition(), mask);
        }
        return Matrix4x4.TRS(root_position, root_rotation, new Vector3(1f, 1f, 1f));
    }


    public void InitPoses()
    {
        int goalStart = 0;
        int goalEnd = 81;
        int inputLength = 81;

        if (PoseInitPath==""){
            FinalActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        } else{
            Debug.Log("Init poses from file " + PoseInitPath);
            float[] initData = FileUtility.ReadNthLineFromTextFile(PoseInitPath, inputLength, 0); 
            float[] goalPose = initData[goalStart..goalEnd]; 

            for (int j = 0; j < FinalActor.Bones.Length; j++)
            {
                Vector3 bonePosition = new Vector3(goalPose[j * 3], goalPose[j * 3 + 1], goalPose[j * 3 + 2]);
                FinalActor.Bones[j].Transform.position = bonePosition;
            }
        }
        
        Vector3 rootPosition = GetHighLevelTargetPosition();
        StartRadius = QuantEvalUtil.sample_distance();
        StartAngle = QuantEvalUtil.sample_angle();
        Quaternion rootRotation = Quaternion.LookRotation(new Vector3(-1f, 0f, 0f), Vector3.up);
        rootRotation  *= Quaternion.Euler(Vector3.up * StartAngle);
        QuantEvalUtil.SetActorPoseFromTrainingFile(Actor);
        Actor.transform.SetPositionAndRotation(rootPosition - rootRotation.GetForward().normalized * StartRadius, rootRotation);
        Debug.Log("Actor root position is " + Actor.transform.GetWorldMatrix().GetPosition().ToString());
    }

 
    public Actor GetClosestFinalActor(Transform pivot, String GoalPoseAction) {
		Actor[] actors = GameObject.FindObjectsOfType<Actor>();

		if(actors.Length == 0) {
			return null;
		} else {
			List<Actor> candidates = new List<Actor>();
			for(int i=0; i< actors.Length; i++) {
                if (GoalPoseAction == "Sit" && actors[i].gameObject.layer == 6 && actors[i].gameObject.activeSelf){
                    candidates.Add(actors[i]);
                } else if(GoalPoseAction == "Lie" && actors[i].gameObject.layer == 7 && actors[i].gameObject.activeSelf){
                    candidates.Add(actors[i]);
                }

			}
			if(candidates.Count == 0) {
                Debug.Log("no candidated found, return null!");
				return null;
			}
			Actor closest = candidates[0];
			for(int i=1; i<candidates.Count; i++) {
				if(Vector3.Distance(pivot.position, candidates[i].GetBoneTransformation("Hips").GetPosition()) < Vector3.Distance(pivot.position, closest.GetBoneTransformation("Hips").GetPosition())) {
					closest = candidates[i];
				}
			}
			return closest;
		}
	}


    private void Reset(){
        WalkComplete = false;
        IdleComplete = false;
        FrameCount = 0;
        IdleCount = 0;
        if (BatchEvalObjCurrentCount < BatchEvalObjTotalCount){
            if (BatchEvalObjCurrentCount != 0){
                BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount-1).gameObject.SetActive(false);
            }
            BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount).gameObject.SetActive(true);
            

            Debug.Log("new object name is " + BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount).name);
            string obj_id_obj = BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount).name;
            obj_id_obj = obj_id_obj.Replace("_mesh_world", "");
            PoseInitPath = "Assets/ObjectDemoFinalPoses/" + obj_id_obj + "/final_pose.txt";
            BatchEvalObjCurrentCount += 1;
        } else{
            Debug.Log("Done with all objects!");
            // pause the editor application
            UnityEditor.EditorApplication.isPaused = true;
            return;
        }

        TimeSeriesSparse = new TimeSeries(6, 6, 1f, 1f, 5);
        RootSeries = new TimeSeries.Root(TimeSeriesSparse);
        StyleSeries = new TimeSeries.Style(TimeSeriesSparse,"Idle", "Walk", "Lie", "Sit");
        GoalSeries = new TimeSeries.Goal(TimeSeriesSparse, Controller.GetSignalNames());
        ContactSeries = new TimeSeries.Contact(TimeSeriesSparse, "LeftFoot", "RightFoot");
        PhaseSeries = new TimeSeries.Phase(TimeSeriesSparse);


        RootPrediction = new Matrix4x4[7];
        GoalPrediction = new Matrix4x4[7];
        positions = new Vector3[Actor.Bones.Length];
   
        InitPoses();

        for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
        {
            RootSeries.Transformations[i] = Actor.transform.GetWorldMatrix(true);
            if (StyleSeries.Styles.Length > 0)
            {
                StyleSeries.Values[i][0] = 1f;
                StyleSeries.Values[i][1] = 0f;
                StyleSeries.Values[i][2] = 0f;
                StyleSeries.Values[i][3] = 0f;
            }
            if (GoalSeries.Actions.Length > 0)
            {
                GoalSeries.Values[i][0] = 1f;
                GoalSeries.Values[i][0] = 0f;
                GoalSeries.Values[i][0] = 0f;
                GoalSeries.Values[i][0] = 0f;
            }
            GoalSeries.Transformations[i] = Actor.transform.GetWorldMatrix(true);
            PhaseSeries.Values[i] = Mathf.Repeat((float)i / GetFramerate(), 1f);
        }

        if (SavePositionError){
            hand_list = new float[TotalFrames];
            all_joint_list = new float[TotalFrames];
            hip_list = new float[TotalFrames];
        }
    }

    protected override void Setup()
    {
        BatchEvalObjTotalCount = BatchEvalCollection.transform.childCount;
        foreach (Transform child in BatchEvalCollection.transform)
        {
            child.gameObject.SetActive(false);
        }
        

        if (HighLevel){
            // set the first child to be visible
            BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount).gameObject.SetActive(true);
            string obj_id_obj = BatchEvalCollection.transform.GetChild(BatchEvalObjCurrentCount).name;
            obj_id_obj = obj_id_obj.Replace("_mesh_world", "");
            PoseInitPath = "Assets/ObjectDemoFinalPoses/" + obj_id_obj + "/final_pose.txt";
            Debug.Log("new pose int path is " + PoseInitPath);
            BatchEvalObjCurrentCount += 1;
            FinalActor.gameObject.SetActive(true);
        } else{
            FinalActor.gameObject.SetActive(false);
        }

       	Forward = (Texture)Resources.Load("Forward");
        Left = (Texture)Resources.Load("Left");
        Right = (Texture)Resources.Load("Right");
        Back = (Texture)Resources.Load("Back");
        TurnLeft = (Texture)Resources.Load("TurnLeft");
        TurnRight = (Texture)Resources.Load("TurnRight");
        Disc = (Texture)Resources.Load("Disc");
        SitUI = (Texture)Resources.Load("HumanSit");
        LieUI = (Texture)Resources.Load("HumanLie");
        

        Controller = new Controller();
        Controller.Signal idle = Controller.AddSignal("Idle");
        idle.Default = true;
        idle.Velocity = 0f;
        idle.AddKey(KeyCode.W, false);
        idle.AddKey(KeyCode.A, false);
        idle.AddKey(KeyCode.S, false);
        idle.AddKey(KeyCode.D, false);
        idle.AddKey(KeyCode.Q, false);
        idle.AddKey(KeyCode.E, false);
        idle.AddKey(KeyCode.C, false);
        idle.AddKey(KeyCode.L, false);
        idle.UserControl = 0.25f;
        idle.NetworkControl = 0.1f;

        Controller.Signal walk = Controller.AddSignal("Walk");
        walk.AddKey(KeyCode.W, true);
        walk.AddKey(KeyCode.A, true);
        walk.AddKey(KeyCode.S, true);
        walk.AddKey(KeyCode.D, true);
        walk.AddKey(KeyCode.Q, true);
        walk.AddKey(KeyCode.E, true);
        walk.AddKey(KeyCode.C, false);
        walk.AddKey(KeyCode.L, false);
        walk.Velocity = 1f;
        walk.UserControl = 0.25f;
        walk.NetworkControl = 0.0f;

        Controller.Signal lie = Controller.AddSignal("Lie");
        lie.AddKey(KeyCode.W, false);
        lie.AddKey(KeyCode.A, false);
        lie.AddKey(KeyCode.S, false);
        lie.AddKey(KeyCode.D, false);
        lie.AddKey(KeyCode.Q, false);
        lie.AddKey(KeyCode.E, false);
        lie.AddKey(KeyCode.C, false);
        lie.AddKey(KeyCode.L, true);
        lie.Velocity = 0f;
        lie.UserControl = 0.25f;
        lie.NetworkControl = 0.0f;

        Controller.Signal sit = Controller.AddSignal("Sit");
        sit.AddKey(KeyCode.W, false);
        sit.AddKey(KeyCode.A, false);
        sit.AddKey(KeyCode.S, false);
        sit.AddKey(KeyCode.D, false);
        sit.AddKey(KeyCode.Q, false);
        sit.AddKey(KeyCode.E, false);
        sit.AddKey(KeyCode.L, false);
        sit.AddKey(KeyCode.C, true);
        sit.Velocity = 0f;
        sit.UserControl = 0.25f;
        sit.NetworkControl = 0.0f;


        TimeSeriesSparse = new TimeSeries(6, 6, 1f, 1f, 5);
        RootSeries = new TimeSeries.Root(TimeSeriesSparse);
        StyleSeries = new TimeSeries.Style(TimeSeriesSparse,"Idle", "Walk", "Lie", "Sit");
        GoalSeries = new TimeSeries.Goal(TimeSeriesSparse, Controller.GetSignalNames());
        ContactSeries = new TimeSeries.Contact(TimeSeriesSparse, "LeftFoot", "RightFoot");
        PhaseSeries = new TimeSeries.Phase(TimeSeriesSparse);

  
        RootPrediction = new Matrix4x4[7];
        GoalPrediction = new Matrix4x4[7];
        positions = new Vector3[Actor.Bones.Length];
   
        InitPoses();

        for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
        {
            RootSeries.Transformations[i] = Actor.transform.GetWorldMatrix(true);
            if (StyleSeries.Styles.Length > 0)
            {
                StyleSeries.Values[i][0] = 1f;
                StyleSeries.Values[i][1] = 0f;
                StyleSeries.Values[i][2] = 0f;
                StyleSeries.Values[i][3] = 0f;
            }
            if (GoalSeries.Actions.Length > 0)
            {
                GoalSeries.Values[i][0] = 1f;
                GoalSeries.Values[i][0] = 0f;
                GoalSeries.Values[i][0] = 0f;
                GoalSeries.Values[i][0] = 0f;
            }
            GoalSeries.Transformations[i] = Actor.transform.GetWorldMatrix(true);
            PhaseSeries.Values[i] = Mathf.Repeat((float)i / GetFramerate(), 1f);
        }
        Debug.Log("Framerate is " + GetFramerate().ToString());
        RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("RightUpLeg"), Actor.GetBoneTransforms("RightFoot"));
        LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("LeftUpLeg"), Actor.GetBoneTransforms("LeftFoot"));
        RightHandIK = UltimateIK.BuildModel(Actor.FindTransform("RightShoulder"), Actor.GetBoneTransforms("RightHand"));
        LeftHandIK = UltimateIK.BuildModel(Actor.FindTransform("LeftShoulder"), Actor.GetBoneTransforms("LeftHand"));
        SpineIK = UltimateIK.BuildModel(Actor.FindTransform("Hips"), Actor.GetBoneTransforms("Head"));

        if (SavePositionError){
            hand_list = new float[TotalFrames];
            all_joint_list = new float[TotalFrames];
            hip_list = new float[TotalFrames];
        }
    }



    protected override void Feed()
    {
        Controller.Update();
        if (!HighLevel){
            FinalActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        }
        
        Signals = Controller.PoolSignals();
        UserControl = Controller.PoolUserControl(Signals);
        NetworkControl = Controller.PoolNetworkControl(Signals);
        ActivateGoal = Controller.QuerySignal("Sit") || Controller.QuerySignal("Lie") || (HighLevel && WalkComplete);
        GoalPoseAction = Controller.QuerySignal("Sit") ? "Sit" : Controller.QuerySignal("Lie") ? "Lie" : GoalPoseAction;

        if (IsInteracting)
		{
            if (HighLevel && !WalkComplete)
            {
                for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
                {
                    GoalSeries.Values[i][0] = 0f;
                    GoalSeries.Values[i][1] = 1f;   
                    GoalSeries.Values[i][2] = 0f;
                    GoalSeries.Values[i][3] = 0f;
                }

            } 
            else if(HighLevel && WalkComplete && GoalPoseAction.Contains("Sit") && !SitComplete){
                for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
                {
                    GoalSeries.Values[i][0] = 0f;
                    GoalSeries.Values[i][1] = 0f;   
                    GoalSeries.Values[i][2] = 0f;
                    GoalSeries.Values[i][3] = 1f;
               }
            } else if(HighLevel && WalkComplete && GoalPoseAction.Contains("Lie") && !LieComplete){
                for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
                {
                    GoalSeries.Values[i][0] = 0f;
                    GoalSeries.Values[i][1] = 0f;   
                    GoalSeries.Values[i][2] = 1f;
                    GoalSeries.Values[i][3] = 0f;
               }
            }

		} else if (HighLevel && !IdleComplete){
            for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
            {
                GoalSeries.Values[i][0] = 1f;
                GoalSeries.Values[i][1] = 0f;   
                GoalSeries.Values[i][2] = 0f;
                GoalSeries.Values[i][3] = 0f;
                StyleSeries.Values[i][0] = 1f;
                StyleSeries.Values[i][1] = 0f;
                StyleSeries.Values[i][2] = 0f;
                StyleSeries.Values[i][3] = 0f;
            }
            IdleCount += 1;
            if (IdleCount > 40){
                IdleComplete = true;
                Debug.Log("Idile is complete!");
            }

        }
        else if((HighLevel && !WalkComplete && IdleComplete) || (HighLevel && WalkComplete && SitComplete)){
            if(HighLevel && WalkComplete && SitComplete){
                Debug.Log("Sit complete! Stand up!");
                for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
                {
                    GoalSeries.Values[i][0] = 0f;
                    GoalSeries.Values[i][1] = 1f;   
                    GoalSeries.Values[i][2] = 0f;
                    GoalSeries.Values[i][3] = 0f;
                }
            }
            StartCoroutine(WalkHighLevelGoal());
        }
		else if (Controller.QuerySignal("Sit") || (HighLevel && WalkComplete && GoalPoseAction.Contains("Sit")))
		{
			StartCoroutine(Sit());
		}		
        else if (Controller.QuerySignal("Lie") || (HighLevel && WalkComplete && GoalPoseAction.Contains("Lie")))
		{
			StartCoroutine(Lie());
		}
		else
		{
			Default();
		}

        //Get Root
        Matrix4x4 root = RootSeries.Transformations[TimeSeriesSparse.Pivot];

        //Input Bone Positions / Velocities
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));     
            NeuralNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
        }



        //Input Trajectory Positions / Directions / Velocities / Styles
        for (int i = 0; i < TimeSeriesSparse.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeriesSparse.GetKey(i);

            NeuralNetwork.FeedXZ(RootSeries.GetPosition(sample.Index).GetRelativePositionTo(root));
            NeuralNetwork.FeedXZ(RootSeries.GetDirection(sample.Index).GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(StyleSeries.Values[sample.Index]);
        }


        // Goal Pose
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            if (ActivateGoal){
                NeuralNetwork.Feed(FinalActor.Bones[i].Transform.position.GetRelativePositionTo(root));
            } else{
                NeuralNetwork.Feed(new Vector3(0f, 0f, 0f));
            }
        }


        //Input Goals
        for (int i = 0; i < TimeSeriesSparse.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeriesSparse.GetKey(i);
            NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetPosition().GetRelativePositionTo(root));
            NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetForward().GetRelativeDirectionTo(root));
            NeuralNetwork.Feed(GoalSeries.Values[sample.Index]);
        }

        //Setup Gating Features
        NeuralNetwork.Feed(GenerateGating());
    }


    protected override void Read()
    {


        //Update Past State
        for (int i = 0; i < TimeSeriesSparse.Pivot; i++)
        {
            TimeSeries.Sample sample = TimeSeriesSparse.Samples[i];
            PhaseSeries.Values[i] = PhaseSeries.Values[i + 1];
            RootSeries.SetPosition(i, RootSeries.GetPosition(i + 1));
            RootSeries.SetDirection(i, RootSeries.GetDirection(i + 1));
            for (int j = 0; j < StyleSeries.Styles.Length; j++)
            {
                StyleSeries.Values[i][j] = StyleSeries.Values[i + 1][j];
            }
            for (int j = 0; j < ContactSeries.Bones.Length; j++)
            {
                ContactSeries.Values[i][j] = ContactSeries.Values[i + 1][j];
            }
            GoalSeries.Transformations[i] = GoalSeries.Transformations[i + 1];
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                GoalSeries.Values[i][j] = GoalSeries.Values[i + 1][j];
            }


        }

        //Get Root
        Matrix4x4 root = RootSeries.Transformations[TimeSeriesSparse.Pivot];

        //Read Posture
        Vector3[] forwards = new Vector3[Actor.Bones.Length];
        Vector3[] upwards = new Vector3[Actor.Bones.Length];
        Vector3[] velocities = new Vector3[Actor.Bones.Length];

        for (int i = 0; i < Actor.Bones.Length; i++)
        {

            Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
            Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
            positions[i] = Vector3.Lerp(Actor.Bones[i].Transform.position + velocity / GetFramerate(), position, 0.5f);
            forwards[i] = forward;
            upwards[i] = upward;
            velocities[i] = velocity;
        }
   
        Matrix4x4 GoalRoot = GetGoalRoot();

        float avg_weight = 0f; 
        if (InverseBlend && ActivateGoal){
            float total_distance = GetPoseToGoalDistance()/ (float)Actor.GetBoneNames().Length;

            avg_weight = GetBlendingWeights(total_distance);

            if (Controller.QuerySignal("Sit") || (HighLevel && GoalPoseAction.Contains("Sit"))){
                avg_weight *= StyleSeries.Values[TimeSeriesSparse.Pivot][3];
            }
            if (Controller.QuerySignal("Lie") || (HighLevel && GoalPoseAction.Contains("Lie"))){
                avg_weight *= StyleSeries.Values[TimeSeriesSparse.Pivot][2];
            }

   
        }

        //Read Inverse Pose
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            string BoneName = Actor.Bones[i].GetName();
            if (ActivateGoal){
                GoalRoot = Matrix4x4.TRS(FinalActor.GetBoneTransformation(BoneName).GetPosition(), GoalRoot.GetRotation(), new Vector3(1f, 1f, 1f));
            } else{
                GoalRoot = RootSeries.Transformations.Last();
            }

            InverseActor.Bones[i].Transform.position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(GoalRoot);
            Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(GoalRoot);
            Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(GoalRoot);
            InverseActor.Bones[i].Transform.rotation = Quaternion.LookRotation(forward, upward);
            InverseActor.Bones[i].ApplyLength();
            Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(GoalRoot);
            if (ActivateGoal && InverseBlend){
                positions[i] = Vector3.Lerp(positions[i], InverseActor.Bones[i].Transform.position, avg_weight);
            }
            else{
                velocities[i] = Vector3.Lerp(velocities[i], GetFramerate() * (InverseActor.Bones[i].Transform.position - Actor.Bones[i].Transform.position), 1f / GetFramerate());
            }
        }



        //Read Future Trajectory
        for (int i = TimeSeriesSparse.PivotKey; i < TimeSeriesSparse.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeriesSparse.GetKey(i);
            Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(root);
            Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root);
            RootSeries.SetPosition(sample.Index, pos);
            RootSeries.SetDirection(sample.Index, dir);
            float[] styles = NeuralNetwork.Read(StyleSeries.Styles.Length);
            for (int j = 0; j < styles.Length; j++)
            {
                styles[j] = Mathf.Clamp(styles[j], 0f, 1f);
            }
            StyleSeries.Values[sample.Index] = styles;

            RootPrediction[i - 6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
        }

        //Read Inverse Trajectory
        for (int i = TimeSeriesSparse.PivotKey; i < TimeSeriesSparse.KeyCount; i++)
        {
            TimeSeries.Sample sample = TimeSeriesSparse.GetKey(i);
            Matrix4x4 goal = GoalSeries.Transformations[TimeSeriesSparse.Pivot];
            goal[1, 3] = 0f;
            Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(goal);
            Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(goal);
            if (i > TimeSeriesSparse.PivotKey)
            {
                Matrix4x4 pivot = RootSeries.Transformations[sample.Index];
                pivot[1, 3] = 0f;
                Matrix4x4 reference = GoalSeries.Transformations[sample.Index];
                reference[1, 3] = 0f;
                float distance = Vector3.Distance(pivot.GetPosition(), reference.GetPosition());
                float weight = Mathf.Pow((float)(i - 6) / 7f, distance * distance);

                RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(sample.Index), pos, weight));
                RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(sample.Index), dir, weight));
            }

            GoalPrediction[i - 6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
        }

        //Read and Correct Goals
        for (int i = 0; i < TimeSeriesSparse.KeyCount; i++)
        {
            float weight = TimeSeriesSparse.GetWeight1byN1(TimeSeriesSparse.GetKey(i).Index, 2f);
            TimeSeries.Sample sample = TimeSeriesSparse.GetKey(i);
            Vector3 pos = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
            Vector3 dir = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
            float[] actions = NeuralNetwork.Read(GoalSeries.Actions.Length);
            for (int j = 0; j < actions.Length; j++)
            {
                actions[j] = Mathf.Clamp(actions[j], 0f, 1f);
            }
            GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[sample.Index], Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one), weight * NetworkControl);
            GoalSeries.Values[sample.Index] = Utility.Interpolate(GoalSeries.Values[sample.Index], actions, weight * NetworkControl);
        }

        //Read Future Contacts
        float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length);
        for (int i = 0; i < contacts.Length; i++)
        {
            contacts[i] = Mathf.Clamp(contacts[i], 0f, 1f);
        }
        ContactSeries.Values[TimeSeriesSparse.Pivot] = contacts;

        //Read Phase Update
        float phase = PhaseSeries.Values[TimeSeriesSparse.Pivot];
        for (int i = TimeSeriesSparse.PivotKey; i < TimeSeriesSparse.KeyCount; i++)
        {
            PhaseSeries.Values[TimeSeriesSparse.GetKey(i).Index] = Mathf.Repeat(phase + NeuralNetwork.Read(), 1f);
        }

        //Interpolate Current to Future Trajectory
        for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
        {
            float weight = (float)(i % TimeSeriesSparse.Resolution) / TimeSeriesSparse.Resolution;
            TimeSeries.Sample sample = TimeSeriesSparse.Samples[i];
            TimeSeries.Sample prevSample = TimeSeriesSparse.GetPreviousKey(i);
            TimeSeries.Sample nextSample = TimeSeriesSparse.GetNextKey(i);

            RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(prevSample.Index), RootSeries.GetPosition(nextSample.Index), weight));
            RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(prevSample.Index), RootSeries.GetDirection(nextSample.Index), weight));
            GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[prevSample.Index], GoalSeries.Transformations[nextSample.Index], weight);
            for (int j = 0; j < StyleSeries.Styles.Length; j++)
            {
                StyleSeries.Values[i][j] = Mathf.Lerp(StyleSeries.Values[prevSample.Index][j], StyleSeries.Values[nextSample.Index][j], weight);
            }
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                GoalSeries.Values[i][j] = Mathf.Lerp(GoalSeries.Values[prevSample.Index][j], GoalSeries.Values[nextSample.Index][j], weight);
            }
        }


        //Assign Posture
        Actor.transform.position = RootSeries.GetPosition(TimeSeriesSparse.Pivot);
		Actor.transform.rotation = RootSeries.GetRotation(TimeSeriesSparse.Pivot);
		for (int i = 0; i < Actor.Bones.Length; i++)
		{
            String BoneName = Actor.Bones[i].GetName();
			Actor.Bones[i].Velocity = velocities[i];
			Actor.Bones[i].Transform.position = positions[i];
			Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
			Actor.Bones[i].ApplyLength();

            
		}

        //Assign Display Actor Posture
        DisplayActor.transform.position = RootSeries.GetPosition(TimeSeriesSparse.Pivot);
		DisplayActor.transform.rotation = RootSeries.GetRotation(TimeSeriesSparse.Pivot);
        // For visualization with rigged character: align rotation
		for (int i = 0; i < DisplayActor.Bones.Length; i++)
		{   
            String BoneName = DisplayActor.Bones[i].GetName();
            if (BoneName == "Hips"){
                DisplayActor.Bones[i].Transform.position = Actor.Bones[i].Transform.position;
                DisplayActor.Bones[i].Transform.rotation = Actor.Bones[i].Transform.rotation;
            } else{
                Quaternion NNrot = Actor.GetBoneTransformation(BoneName).rotation;
                Vector3 NNpos = Actor.GetBoneTransformation(BoneName).GetPosition();
                Quaternion parentDisplayRot = DisplayActor.Bones[i].GetParent().Transform.rotation;
                Vector3 displayPos = DisplayActor.Bones[i].Transform.position;
                // Add a rotation correction for parent
                if (DisplayActor.Bones[i].GetParent().GetName() != "Hips"){
                    Vector3 parentPos = DisplayActor.Bones[i].GetParent().Transform.position;
                    Quaternion correction = Quaternion.FromToRotation((displayPos-parentPos).normalized, (NNpos-parentPos).normalized);
                    DisplayActor.Bones[i].GetParent().Transform.rotation = correction * parentDisplayRot;
                }

                DisplayActor.Bones[i].Transform.position = NNpos;
                DisplayActor.Bones[i].Transform.rotation = NNrot;
            }
		}

        if (FrameCount <= TotalFrames){     
            if (FrameCount == TotalFrames && HighLevel){
                if (SavePositionError){
                    QuantEvalUtil.SaveToFile(ExportPath, hand_list, "object" + BatchEvalObjCurrentCount.ToString() + "_hand_position_error.txt");
                    QuantEvalUtil.SaveToFile(ExportPath, all_joint_list, "object" + BatchEvalObjCurrentCount.ToString() + "_joint_position_error.txt");
                    QuantEvalUtil.SaveToFile(ExportPath, hip_list, "object" + BatchEvalObjCurrentCount.ToString() + "_hip_position_error.txt");
                }
                Debug.Log("Reset!!");
                Reset();
            }else{
                if (SavePositionError){
                    float hand_avg = GetAvgHandToGoalDistance();
                    float all_joint_avg = GetPoseToGoalDistance() / (float)Actor.Bones.Length;
                    hip_list[FrameCount] = GetHipToGoalDistance();
                    hand_list[FrameCount] = hand_avg;
                    all_joint_list[FrameCount] = all_joint_avg;
                }
                FrameCount += 1;
            }    
        }
    }

	private float[] GenerateGating()
	{
		List<float> values = new List<float>();

		for (int k = 0; k < TimeSeriesSparse.KeyCount; k++)
		{
			int index = TimeSeriesSparse.GetKey(k).Index;
			Vector2 phase = Utility.PhaseVector(PhaseSeries.Values[index]);
			for (int i = 0; i < StyleSeries.Styles.Length; i++)
			{
				float magnitude = StyleSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f, 1f);
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
			}
			for (int i = 0; i < GoalSeries.Actions.Length; i++)
			{
				float magnitude = GoalSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f, 1f);
				Matrix4x4 root = RootSeries.Transformations[index];
				root[1, 3] = 0f;
				Matrix4x4 goal = GoalSeries.Transformations[index];
				goal[1, 3] = 0f;
				float distance = Vector3.Distance(root.GetPosition(), goal.GetPosition());
				float angle = Quaternion.Angle(root.GetRotation(), goal.GetRotation());
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
				values.Add(magnitude * distance * phase.x);
				values.Add(magnitude * distance * phase.y);
				values.Add(magnitude * angle * phase.x);
				values.Add(magnitude * angle * phase.y);
			}
		}
		return values.ToArray();
	}


    private void Default()
    {
        if (Controller.ProjectionActive)
        {
            // high level goal
            Debug.Log("Apply Static Goal because projection is activate");
            ApplyStaticGoal(Controller.Projection.point, Vector3.ProjectOnPlane(Controller.Projection.point - transform.position, Vector3.up).normalized, Signals);
        } 
        else
        {
            ApplyDynamicGoal(
                RootSeries.Transformations[TimeSeriesSparse.Pivot],
                Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
                Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f),
                Signals
            );
        }
    }


    private void ApplyStaticGoal(Vector3 position, Vector3 direction, float[] actions)
    {

        for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
        {
            float weight = TimeSeriesSparse.GetWeight1byN1(i, 2f);
            float positionBlending = weight * UserControl;
            float directionBlending = weight * UserControl;
            Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), position, positionBlending));
            Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.LookRotation(Vector3.Slerp(GoalSeries.Transformations[i].GetForward(), direction, directionBlending), Vector3.up));
        }

        //Actions
        for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
        {
            float w = (float)(i - TimeSeriesSparse.Pivot) / (float)(TimeSeriesSparse.FutureSampleCount);
            w = Utility.Normalise(w, 0f, 1f, 1f / TimeSeriesSparse.FutureKeyCount, 1f);
            for (int j = 0; j < GoalSeries.Actions.Length; j++)
            {
                float weight = GoalSeries.Values[i][j];
                weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
                weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f - UserControl);
                if (actions[j] != GoalSeries.Values[i][j])
                {
                    GoalSeries.Values[i][j] = Mathf.Lerp(
                        GoalSeries.Values[i][j],
                        Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
                        w);
                }
            }
        }
    }

	private void ApplyDynamicGoal(Matrix4x4 root, Vector3 move, float turn, float[] actions)
	{
		//Transformations
		Vector3[] positions_blend = new Vector3[TimeSeriesSparse.Samples.Length];
		Vector3[] directions_blend = new Vector3[TimeSeriesSparse.Samples.Length];
		float time = 2f;
		for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
		{
			float weight = TimeSeriesSparse.GetWeight1byN1(i, 0.5f);
			float bias_pos = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			float bias_dir = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			directions_blend[i] = Quaternion.AngleAxis(bias_dir * turn, Vector3.up) * Vector3.ProjectOnPlane(root.GetForward(), Vector3.up).normalized;
			if (i == 0)
			{
				positions_blend[i] = root.GetPosition() +
					Vector3.Lerp(
					GoalSeries.Transformations[i + 1].GetPosition() - GoalSeries.Transformations[i].GetPosition(),
					time / (TimeSeriesSparse.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			}
			else
			{
				positions_blend[i] = positions_blend[i - 1] +
					Vector3.Lerp(
					GoalSeries.Transformations[i].GetPosition() - GoalSeries.Transformations[i - 1].GetPosition(),
					time / (TimeSeriesSparse.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			}
            positions_blend[i][1] = 0f;
		}
		for (int i = 0; i < TimeSeriesSparse.Samples.Length; i++)
		{
			Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), positions_blend[i], UserControl));
			Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.Slerp(GoalSeries.Transformations[i].GetRotation(), Quaternion.LookRotation(directions_blend[i], Vector3.up), UserControl));
		}

		//Actions
		for (int i = TimeSeriesSparse.Pivot; i < TimeSeriesSparse.Samples.Length; i++)
		{
			float w = (float)(i - TimeSeriesSparse.Pivot) / (float)(TimeSeriesSparse.FutureSampleCount);
			w = Utility.Normalise(w, 0f, 1f, 1f / TimeSeriesSparse.FutureKeyCount, 1f);
			for (int j = 0; j < GoalSeries.Actions.Length; j++)
			{
				float weight = GoalSeries.Values[i][j];
				weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
				weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f - UserControl);
				if (actions[j] != GoalSeries.Values[i][j])
				{
					GoalSeries.Values[i][j] = Mathf.Lerp(
						GoalSeries.Values[i][j],
						Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
						w);
				}
			}
		}
	}



    protected override void OnGUIDerived()
    {
        if (!ShowGUI)
        {
            return;
        }
        if (ShowGoal)
        {
            GoalSeries.GUI();
        }
        if (ShowCurrent)
        {
            StyleSeries.GUI();
        }
        if (ShowPhase)
        {
            PhaseSeries.GUI();
        }
        if (ShowContacts)
        {
            ContactSeries.GUI();
        }
        if (ShowMoevement){
            UltiDraw.DrawGUITexture(new Vector2(0.4f, 0.05f), 0.03f, Disc, Input.GetKey(KeyCode.W) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.4f, 0.05f), 0.03f, Forward, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.365f, 0.05f), 0.03f, Disc, Input.GetKey(KeyCode.Q) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.365f, 0.05f), 0.03f, TurnLeft, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.435f, 0.05f), 0.03f, Disc, Input.GetKey(KeyCode.E) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.435f, 0.05f), 0.03f, TurnRight, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.4f, 0.11f), 0.03f, Disc, Input.GetKey(KeyCode.S) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.4f, 0.11f), 0.03f, Back, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.365f, 0.11f), 0.03f, Disc, Input.GetKey(KeyCode.A) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.365f, 0.11f), 0.03f, Left, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.435f, 0.11f), 0.03f, Disc, Input.GetKey(KeyCode.D) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.435f, 0.11f), 0.03f, Right, UltiDraw.White);
			
            UltiDraw.DrawGUITexture(new Vector2(0.5f, 0.08f), 0.055f, Disc, (Input.GetKey(KeyCode.C) || (HighLevel && WalkComplete && GoalPoseAction.Contains("Sit"))) ? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.5f, 0.08f), 0.04f, SitUI, UltiDraw.White);

			UltiDraw.DrawGUITexture(new Vector2(0.56f, 0.08f), 0.055f, Disc, (Input.GetKey(KeyCode.L) || (HighLevel && WalkComplete && GoalPoseAction.Contains("Lie")))? UltiDraw.Orange : UltiDraw.BlackGrey);
			UltiDraw.DrawGUITexture(new Vector2(0.56f, 0.08f), 0.04f, LieUI, UltiDraw.White);

        }

    }

    protected override void OnRenderObjectDerived()
    {


        if (ShowRoot)
        {
            RootSeries.Draw();
        }
        if (ShowGoal)
        {
            GoalSeries.Draw();
        }
        if (ShowCurrent)
        {
            StyleSeries.Draw();
        }
        if (ShowPhase)
        {
            PhaseSeries.Draw();
        }
        if (ShowContacts)
        {
            ContactSeries.Draw();
        }


        if (ShowBiDirectional)
        {

            UltiDraw.Begin();
   
            for (int i = 0; i < RootPrediction.Length; i++)
            {
                UltiDraw.DrawCircle(RootPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkRed.Darken(0.5f));
                UltiDraw.DrawArrow(RootPrediction[i].GetPosition(), RootPrediction[i].GetPosition() + 0.1f * RootPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkRed);
                if (i < RootPrediction.Length - 1)
                {
                    UltiDraw.DrawLine(RootPrediction[i].GetPosition(), RootPrediction[i + 1].GetPosition(), UltiDraw.Black);
                }
            }
            for (int i = 0; i < GoalPrediction.Length; i++)
            {
                UltiDraw.DrawCircle(GoalPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkGreen.Darken(0.5f));
                UltiDraw.DrawArrow(GoalPrediction[i].GetPosition(), GoalPrediction[i].GetPosition() + 0.1f * GoalPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkGreen);
                if (i < GoalPrediction.Length - 1)
                {
                    UltiDraw.DrawLine(GoalPrediction[i].GetPosition(), GoalPrediction[i + 1].GetPosition(), UltiDraw.Black);
                }
            }
            UltiDraw.End();

        }


    }

    float GetBlendingWeights(float distance, float steep=20f, float shift=0.25f)
    {
       return -1f / (1f + (float)Math.Exp(steep * (-(distance - shift)))) + 1f;
    }

    

    protected override void Postprocess()
    {
        Matrix4x4 rightFoot = Actor.GetBoneTransformation(ContactSeries.Bones[1]);
        Matrix4x4 leftFoot = Actor.GetBoneTransformation(ContactSeries.Bones[0]);
        float rightWeight = Math.Max(1f, 1 - ContactSeries.Values[TimeSeriesSparse.Pivot][1]);
        float leftWeight = Math.Max(1f, 1 - ContactSeries.Values[TimeSeriesSparse.Pivot][0]);

        RightFootIK.Objectives[0].SetTarget(rightFoot.GetPosition(), rightWeight);
        RightFootIK.Objectives[0].SetTarget(rightFoot.GetRotation());

        LeftFootIK.Objectives[0].SetTarget(leftFoot.GetPosition(), leftWeight);
        LeftFootIK.Objectives[0].SetTarget(leftFoot.GetRotation());
        float avg_weight =   0f;
        if (Controller.QuerySignal("Sit") || (HighLevel && GoalPoseAction.Contains("Sit"))){
                avg_weight = StyleSeries.Values[TimeSeriesSparse.Pivot][3];
            }
        if (Controller.QuerySignal("Lie") || (HighLevel && GoalPoseAction.Contains("Lie"))){
            avg_weight = StyleSeries.Values[TimeSeriesSparse.Pivot][2];
        }


        RightHandIK.Objectives[0].SetTarget(Vector3.Lerp(Actor.GetBoneTransformation("RightHand").GetPosition(), InverseActor.GetBoneTransformation("RightHand").GetPosition(), avg_weight), 1f);
        RightHandIK.Objectives[0].SetTarget(Quaternion.Slerp(Actor.GetBoneTransformation("RightHand").GetRotation(), InverseActor.GetBoneTransformation("RightHand").GetRotation(), avg_weight));

        LeftHandIK.Objectives[0].SetTarget(Actor.GetBoneTransformation("LeftHand").GetPosition(), 1f);
        LeftHandIK.Objectives[0].SetTarget(Actor.GetBoneTransformation("LeftHand").GetRotation());

        RightFootIK.Solve();
        LeftFootIK.Solve();
    }

    private float GetPoseToGoalDistance(){
        float total_dist = 0f;
        for (int i = 0; i < Actor.Bones.Length; i++)
        {
            Vector3 curr_pos = Actor.Bones[i].Transform.position;
            float distance = Vector3.Distance(FinalActor.Bones[i].Transform.position, curr_pos);
            total_dist += distance;
        }
        return total_dist;
    }

     private float GetAvgHandToGoalDistance(){
        float dist1 = Vector3.Distance(Actor.GetBoneTransformation("LeftHand").GetPosition(), FinalActor.GetBoneTransformation("LeftHand").GetPosition());
        float dist2 = Vector3.Distance(Actor.GetBoneTransformation("RightHand").GetPosition(), FinalActor.GetBoneTransformation("RightHand").GetPosition());
        return (dist1 + dist2)/2;
    }

     private float GetHipToGoalDistance(){
        float dist1 = Vector3.Distance(Actor.GetBoneTransformation("Hips").GetPosition(), FinalActor.GetBoneTransformation("Hips").GetPosition());
        return dist1;
    }

    private IEnumerator Sit()
	{
		Controller.Signal signal = Controller.GetSignal("Sit");
        if (!HighLevel){
            FinalActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        }
        
        IsInteracting = true;
        while (signal.Query() || (HighLevel && !SitComplete && IdleComplete))
        {
            if (HighLevel && !SitComplete){
                Signals[0] = 0f;
                Signals[3] = 1f;
            }
            ApplyStaticGoal(GetHighLevelTargetPosition(), GetHighLevelTargetForward(), Signals);
            if(GetPoseToGoalDistance() < 0.5f){
                break;
            }
            yield return new WaitForSeconds(0f);
        }
        Controller.ActiveInteraction = null;
        IsInteracting = false;
	}


    private IEnumerator Lie() 
	{
        if (!HighLevel){
            FinalActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        }
        
		Controller.Signal signal = Controller.GetSignal("Lie");
        Debug.Log("entered lie co-routine");
        IsInteracting = true;
        while (signal.Query() || HighLevel)
        {
            if (HighLevel && ! SitComplete){
                // idle, walk, lie sit
                Signals[0] = 0f;
                Signals[2] = 1f;
            }
            bool enableHit = false;
            ApplyStaticGoal(GetHighLevelTargetPosition(), GetHighLevelTargetForward(enableHit), Signals);
            yield return new WaitForSeconds(0f);
        }

        Controller.ActiveInteraction = null;
        IsInteracting = false;
	}

    private IEnumerator WalkHighLevelGoal()
    {
        IsInteracting = true;
        Debug.Log("Entered high level walk co-routine"); 
        Vector3 targetPosition = GetHighLevelTargetPosition();
        Vector3 targetForward = GetHighLevelTargetForward();
        float distance = Vector3.Distance(Actor.transform.GetWorldMatrix().GetPosition(), targetPosition);    
        while (distance > FootDistanceThreshold) 
        {
            if (distance > FootDistanceThreshold + 2f){
                Vector3 targetForwardFlip = targetForward;
                targetForwardFlip[0] = -targetForward[0];
                targetForwardFlip[2] = -targetForward[2];
                ApplyStaticGoal(targetPosition, targetForwardFlip, Signals);
            }else{
                ApplyStaticGoal(targetPosition, targetForward, Signals);
            }  
            distance = Vector3.Distance(Actor.transform.GetWorldMatrix().GetPosition(), targetPosition);
            yield return new WaitForSeconds(0f);
        }
        IsInteracting = false;
        WalkComplete = true;
        Debug.Log("High level walk is complete!");
    }


    private Vector3 GetHighLevelTargetPosition()
    {
        Vector3 GoalPos = FinalActor.GetBoneTransformation("Hips").GetPosition();
        if (!HighLevel){
            GoalPos = GetClosestFinalActor(Actor.transform, GoalPoseAction).GetBoneTransformation("Hips").GetPosition();
        } 
        LayerMask mask = LayerMask.GetMask("Ground");
        return Utility.ProjectGround(GoalPos, mask);
    }

    private Vector3 GetHighLevelTargetForward(bool enableHit=true)
    {
        Actor closeActor = GetClosestFinalActor(Actor.transform, GoalPoseAction);
        if (HighLevel){
            closeActor = FinalActor;
        }
       
        Vector3 v1 = Vector3.ProjectOnPlane(closeActor.GetBoneTransformation("RightUpLeg").GetPosition() - closeActor.GetBoneTransformation("LeftUpLeg").GetPosition(), Vector3.up).normalized;
        Vector3 v2 = Vector3.ProjectOnPlane(closeActor.GetBoneTransformation("RightShoulder").GetPosition() - closeActor.GetBoneTransformation("LeftShoulder").GetPosition(), Vector3.up).normalized;
        Vector3 v = (v1 + v2).normalized;
        Vector3 forward = Vector3.ProjectOnPlane(-Vector3.Cross(v, Vector3.up), Vector3.up).normalized;


        if (enableHit && GoalPoseAction.Contains("Lie")){
            v1 = Vector3.ProjectOnPlane(closeActor.GetBoneTransformation("RightLeg").GetPosition() - closeActor.GetBoneTransformation("LeftLeg").GetPosition(), Vector3.up).normalized;
            v2 = Vector3.ProjectOnPlane(closeActor.GetBoneTransformation("LeftLeg").GetPosition() - closeActor.GetBoneTransformation("RightLeg").GetPosition(), Vector3.up).normalized;
            RaycastHit hit;
            int layerMask = 1 << 8;
            layerMask = ~layerMask; // all except layer 0

            if (Physics.Raycast(closeActor.GetBoneTransformation("Hips").GetPosition(), v1, out hit, Mathf.Infinity, layerMask))
            {
                Debug.Log("Did Hit");
                return v2;
            }
            else
            {
                Debug.Log("Did not Hit");
                return v1;
            }             
        } 
        else{
             return forward;
       }        
    }


}
