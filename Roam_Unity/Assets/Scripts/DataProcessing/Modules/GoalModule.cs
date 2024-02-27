#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;
using System;

public class GoalModule : Module {

	public bool[] Keys = new bool[0];
	public TargetFunction Target = null;
	public GoalFunction[] Functions = new GoalFunction[0];

	public override ID GetID() {
		return ID.Goal;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Target = new TargetFunction(this);
		Functions = new GoalFunction[0];
		Keys = new bool[data.GetTotalFrames()];
		Keys[0] = true;
		Keys[Keys.Length-1] = true;
		return this;
	}

	public override void Slice(Sequence sequence) {
		Debug.Log("Slicing not implemented for " + GetID() + ".");
	}

	public override void Callback(MotionEditor editor) {
		
	}

	public void AddGoal(string name) {
		if(System.Array.Exists(Functions, x => x.Name == name)) {
			Debug.Log("Goal with name " + name + " already exists.");
			return;
		}
		ArrayExtensions.Add(ref Functions, new GoalFunction(this, name));
	}

	public void RemoveGoal(string name) {
		int index = System.Array.FindIndex(Functions, x => x.Name == name);
		if(index >= 0) {
			ArrayExtensions.RemoveAt(ref Functions, index);
		} else {
			Debug.Log("Goal with name " + name + " does not exist.");
		}
	}

	public GoalFunction GetGoalFunction(string name) {
		return System.Array.Find(Functions, x => x.Name == name);
	}

	public float[] GetActions(Frame frame, float delta) {
		//Delta for taking the goal from one frame ahead when being at the current frame.
		float[] actions = new float[Functions.Length];
		for(int i=0; i<actions.Length; i++) {
			actions[i] = Functions[i].GetValue(Data.GetFrame(frame.Timestamp - 1f/Data.Framerate + delta));
		}
		return actions;
	}

	public string[] GetNames() {
		string[] names = new string[Functions.Length];
		for(int i=0; i<names.Length; i++) {
			names[i] = Functions[i].Name;
		}
		return names;
	}

	public void ToggleKey(Frame frame) {
		Keys[frame.Index-1] = !Keys[frame.Index-1];
		Target.Compute(frame);
		for(int i=0; i<Functions.Length; i++) {
			Functions[i].Compute(frame);
		}
	}

	public bool IsKey(Frame frame) {
		return Keys[frame.Index-1];
	}

	public Frame GetPreviousKey(Frame frame) {
		while(frame.Index > 1) {
			frame = Data.GetFrame(frame.Index-1);
			if(IsKey(frame)) {
				return frame;
			}
		}
		return Data.Frames.First();
	}

	public Frame GetNextKey(Frame frame) {
		while(frame.Index < Data.GetTotalFrames()) {
			frame = Data.GetFrame(frame.Index+1);
			if(IsKey(frame)) {
				return frame;
			}
		}
		return Data.Frames.Last();
	}

	protected override void DerivedDraw(MotionEditor editor) {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		Frame frame = editor.GetCurrentFrame();

		if(Utility.GUIButton("Key", IsKey(frame) ? UltiDraw.Cyan : UltiDraw.DarkGrey, IsKey(frame) ? UltiDraw.Black : UltiDraw.White)) {
			ToggleKey(frame);
		}
		if (Utility.GUIButton("Estimate Goal", UltiDraw.DarkGrey, UltiDraw.White))
		{
			EditorCoroutines.StartCoroutine(CaptureGoalTransition(editor), this);
        }
		if (Utility.GUIButton("Compute Transition Count", UltiDraw.DarkGrey, UltiDraw.White))
		{
			EditorCoroutines.StartCoroutine(CaptureTransitionCount(editor), this);
		}
		if (Utility.GUIButton("Estimate Pivot", UltiDraw.DarkGrey, UltiDraw.White))
		{
			EditorCoroutines.StartCoroutine(Target.CapturePivot(editor), this);
		}
		if (Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White))
		{
			for (int f = 0; f < Functions.Length; f++)
			{
				Functions[f].Values = new float[Data.GetTotalFrames()];
			}
			Keys = new bool[Data.GetTotalFrames()];
			Keys[0] = true;
			Keys[Keys.Length - 1] = true;
		}

		EditorGUI.BeginDisabledGroup(!IsKey(frame));
		Color[] colors = UltiDraw.GetRainbowColors(Functions.Length);
		for(int i=0; i<1; i++) {
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Pivot", GUILayout.Width(60f));
			Target.SetPivot(frame, (TargetFunction.Pivot)EditorGUILayout.EnumPopup(Target.GetPivot(frame)));
			EditorGUILayout.LabelField("Reference", GUILayout.Width(60f));
			Target.SetReference(frame, EditorGUILayout.IntField(Target.GetReference(frame)));
			EditorGUILayout.LabelField("Object", GUILayout.Width(60f));
			Target.SetProp(frame, EditorGUILayout.TextField(Target.GetProp(frame)));
			EditorGUILayout.LabelField("Specifier", GUILayout.Width(60f));
			Target.SetSpecifier(frame, EditorGUILayout.TextField(Target.GetSpecifier(frame)));
			EditorGUI.EndDisabledGroup();
			EditorGUILayout.LabelField("Resolution", GUILayout.Width(60f));
			Target.PropResolution = EditorGUILayout.IntField(Target.PropResolution, GUILayout.Width(40f));
			if(Utility.GUIButton("Geometry", Target.DrawGeometry ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 100f, 18f)) {
				Target.DrawGeometry = !Target.DrawGeometry;
			}
			if(Utility.GUIButton("References", Target.DrawReferences ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 100f, 18f)) {
				Target.DrawReferences = !Target.DrawReferences;
			}
			if(Utility.GUIButton("Distribution", Target.DrawDistribution ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 100f, 18)) {
				Target.DrawDistribution = !Target.DrawDistribution;
			}
			EditorGUI.BeginDisabledGroup(!IsKey(frame));
			EditorGUILayout.EndHorizontal();
		}

		for(int i=0; i<Functions.Length; i++) {
			float height = 25f;
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton(Functions[i].Name, colors[i].Transparent(Utility.Normalise(Functions[i].GetValue(frame), 0f, 1f, 0.25f, 1f)), UltiDraw.White, 150f, height)) {
				Functions[i].Toggle(frame);
			}
			Rect c = EditorGUILayout.GetControlRect();
			Rect r = new Rect(c.x, c.y, Functions[i].GetValue(frame) * c.width, height);
			EditorGUI.DrawRect(r, colors[i].Transparent(0.75f));
			EditorGUILayout.FloatField(Functions[i].GetValue(frame), GUILayout.Width(50f));
			EditorGUILayout.IntField(Functions[i].GetTransitionCount(frame), GUILayout.Width(50f));
			Functions[i].Name = EditorGUILayout.TextField(Functions[i].Name);
			if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f)) {
				RemoveGoal(Functions[i].Name);
			}
			EditorGUILayout.EndHorizontal();
		}
		EditorGUI.EndDisabledGroup();

		if(Utility.GUIButton("Add Goal", UltiDraw.DarkGrey, UltiDraw.White)) {
			AddGoal("Goal " + (Functions.Length+1));
			EditorGUIUtility.ExitGUI();
		}
		EditorGUILayout.BeginHorizontal();
		if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			editor.LoadFrame(GetPreviousKey(frame));
		}
		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		UltiDraw.Begin();

		float startTime = frame.Timestamp-editor.GetWindow()/2f;
		float endTime = frame.Timestamp+editor.GetWindow()/2f;
		if(startTime < 0f) {
			endTime -= startTime;
			startTime = 0f;
		}
		if(endTime > Data.GetTotalTime()) {
			startTime -= endTime-Data.GetTotalTime();
			endTime = Data.GetTotalTime();
		}
		startTime = Mathf.Max(0f, startTime);
		endTime = Mathf.Min(Data.GetTotalTime(), endTime);
		int start = Data.GetFrame(startTime).Index;
		int end = Data.GetFrame(endTime).Index;
		int elements = end-start;

		Vector3 prevPos = Vector3.zero;
		Vector3 newPos = Vector3.zero;
		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

		for(int i=0; i<Functions.Length; i++) {
			Frame current = Data.Frames.First();
			while(current != Data.Frames.Last()) {
				Frame next = GetNextKey(current);
				if(Functions[i].Values[current.Index-1] == 1f) {
					float _start = (float)(Mathf.Clamp(current.Index, start, end)-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(next.Index, start, end)-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					Vector3 a = new Vector3(xStart, rect.y, 0f);
					Vector3 b = new Vector3(xEnd, rect.y, 0f);
					Vector3 c = new Vector3(xStart, rect.y+rect.height, 0f);
					Vector3 d = new Vector3(xEnd, rect.y+rect.height, 0f);
					UltiDraw.DrawTriangle(a, c, b, colors[i].Transparent(0.25f));
					UltiDraw.DrawTriangle(b, c, d, colors[i].Transparent(0.25f));
				}
				current = next;
			}
		}

		//Keys
		for(int i=0; i<Keys.Length; i++) {
			if(Keys[i]) {
				top.x = rect.xMin + (float)(i+1-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(i+1-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.White);
			}
		}

		//Current Pivot
		float pStart = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp-1f, 0f, Data.GetTotalTime())).Index-start) / (float)elements;
		float pEnd = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp+1f, 0f, Data.GetTotalTime())).Index-start) / (float)elements;
		float pLeft = rect.x + pStart * rect.width;
		float pRight = rect.x + pEnd * rect.width;
		Vector3 pA = new Vector3(pLeft, rect.y, 0f);
		Vector3 pB = new Vector3(pRight, rect.y, 0f);
		Vector3 pC = new Vector3(pLeft, rect.y+rect.height, 0f);
		Vector3 pD = new Vector3(pRight, rect.y+rect.height, 0f);
		UltiDraw.DrawTriangle(pA, pC, pB, UltiDraw.White.Transparent(0.1f));
		UltiDraw.DrawTriangle(pB, pC, pD, UltiDraw.White.Transparent(0.1f));
		top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

		UltiDraw.End();
		
		EditorGUILayout.EndVertical();
		if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			editor.LoadFrame(GetNextKey(frame));
		}
		EditorGUILayout.EndHorizontal();
	}

	private IEnumerator CaptureGoalTransition(MotionEditor editor)
    {
		StyleModule style = (StyleModule)Data.GetModule(ID.Style);
		// initialize functions and keys
		int prevFrame = 0;
		for (int i = 0; i < Data.Frames.Length; i++)
        {
			if (style.Keys[i])
			{
     
				Keys[i] = true;
				for (int j = prevFrame; j < i; j++){
					for (int f = 0; f < Functions.Length; f++)
					{
                        // this requires style key to be exactly 0 or 1
                        // we use round because sometimes there is error in labelling
						Functions[f].Values[j] = (float)Math.Round(style.Functions[f].Values[i]);
					}
				}
				prevFrame = i;
			}
        }
        // key pruning: remove previous key if values are the same
        int PrevKeyIndex = 0;
        for (int i = 0; i < Data.Frames.Length; i++)
        {
            if (Keys[i])
            {
                bool sameGoal = true;
                for (int f = 0; f < Functions.Length; f++)
                {
                    // this requires style key to be exactly 0 or 1
                    if (Functions[f].Values[i] != Functions[f].Values[PrevKeyIndex])
                    {
                        sameGoal = false;
                    }
                    if (sameGoal)
                    {
                        Keys[i] = false;
                    }
                }

                PrevKeyIndex = i;
            }
        }

        Frame current = editor.GetCurrentFrame();
		System.DateTime time = Utility.GetTimestamp();
		if (Utility.GetElapsedTime(time) > 0.2f)
		{
			time = Utility.GetTimestamp();
			yield return new WaitForSeconds(0f);
		}
		editor.LoadFrame(current);
	}

	private IEnumerator CaptureTransitionCount(MotionEditor editor){
        StyleModule style = (StyleModule)Data.GetModule(ID.Style);
        for (int f=0; f < Data.Frames.Length; f++){
			Frame current = Data.Frames[f];
            Frame nextStyleKey = style.GetNextKey(current);
			for(int i=0; i<Functions.Length; i++) {   
                // transition = 0 if goal action is not sit, or current style is sit
                if (Functions[1].Values[current.Index-1] != 1f || style.Functions[1].Values[current.Index-1] == 1f) 
                {
                    Functions[i].TransitionCounts[current.Index-1] = 0;
                }
                else
                {
                    Functions[i].TransitionCounts[current.Index - 1] = nextStyleKey.Index - current.Index;
                }
                
			}
		}
		
		yield return new WaitForSeconds(0f);

	}


    [System.Serializable]
	public class TargetFunction {

		// Pivots moved to global scope
		public GoalModule Module;
		public int PropResolution;
		public bool DrawGeometry = true;
		public bool DrawReferences = false;
		public bool DrawDistribution = false;
		
		public int[] References;
		public string[] Props;
		public string[] Specifiers;

		public enum Pivot {DynamicRoot, StaticRoot, DynamicObject, StaticObject};
	    public Pivot[] Pivots; 


	    public IEnumerator CapturePivot(MotionEditor editor)
        {
		int interaction_index = -1;
		for (int i = 0; i < Module.Functions.Length; i++){
			if (Module.Functions[i].Name == "Sit" || Module.Functions[i].Name == "Lie"){
				interaction_index = i;
			}

		}
		for (int i = 0; i < Module.Data.Frames.Length; i++)
        {
	        if (interaction_index!= -1){
                float sit = Module.Functions[interaction_index].Values[i];
                if (sit == 1f){
				    Pivots[i] = Pivot.StaticObject;
			    }
			} else{
				Pivots[i] = Pivot.DynamicRoot;
			}
		}

		Frame current = editor.GetCurrentFrame();
		System.DateTime time = Utility.GetTimestamp();

		if (Utility.GetElapsedTime(time) > 0.2f)
		{
			time = Utility.GetTimestamp();
			yield return new WaitForSeconds(0f);
		}

		editor.LoadFrame(current);
	    }

		public void Fix() {
			Specifiers = new string[Module.Data.Frames.Length];
			for(int i=0; i<Specifiers.Length; i++) {
				Specifiers[i] = string.Empty;
			}
		}

		public TargetFunction(GoalModule module) {
			Module = module;
		    Pivots = new Pivot[Module.Data.Frames.Length];
			References = new int[Module.Data.Frames.Length];
			Props = new string[Module.Data.Frames.Length];
			Specifiers = new string[Module.Data.Frames.Length];
			PropResolution = 8;
			for(int i=0; i<Pivots.Length; i++) {
				Pivots[i] = Pivot.DynamicRoot;
			}
			for(int i=0; i<References.Length; i++) {
				References[i] = 25;
			}
			for(int i=0; i<Props.Length; i++) {
				Props[i] = string.Empty;
			}
			for(int i=0; i<Specifiers.Length; i++) {
				Specifiers[i] = string.Empty;
			}
		}

		public Matrix4x4 GetGoalTransformation(Frame reference, float offset, bool mirrored, float delta) {
			return Matrix4x4.TRS(GetGoalPosition(reference, offset, mirrored, delta), GetGoalRotation(reference, offset, mirrored, delta), Vector3.one);
		}
 
 		public Vector3 GetGoalPosition(Frame reference, float offset, bool mirrored, float delta) {
            Frame frame = Module.Data.GetFrame(reference.Timestamp + offset);
            if (GetPivot(frame) == Pivot.StaticObject) {
				// retrieve the goal pose
				Frame nextKeyFrame = Module.GetNextKey(frame);
				Matrix4x4 hip = nextKeyFrame.GetBoneTransformation("Hips", mirrored);	
				Vector3 hipPosition = hip.GetPosition();	
				hipPosition[1] = 0f;
 				return hipPosition;
			}
			else {	
				RootModule module = Module.Data.GetModule(ID.Root) == null ? null : (RootModule)Module.Data.GetModule(ID.Root);
				if(module == null) {
					return Vector3.zero;
				} else {
					return module.GetEstimatedRootPosition(reference, GetTargetTimestamp(frame, delta) - frame.Timestamp + offset, mirrored);
				}
			}
		}

		public Quaternion DeriveRootRotationFrom4Joints(Frame frame, bool mirrored){
			Matrix4x4 hip = frame.GetBoneTransformation("Hips", mirrored);
			Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation("RightUpLeg", mirrored).GetPosition() - frame.GetBoneTransformation("LeftUpLeg", mirrored).GetPosition(), Vector3.up).normalized;
			Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation("RightShoulder", mirrored).GetPosition() - frame.GetBoneTransformation("LeftShoulder", mirrored).GetPosition(), Vector3.up).normalized;
			Vector3 v = (v1+v2).normalized;
			Vector3 forward = Vector3.ProjectOnPlane(-Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
			return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
		}
		

		public Quaternion GetGoalRotation(Frame reference, float offset, bool mirrored, float delta) {
			Frame frame = Module.Data.GetFrame(reference.Timestamp + offset);
            if (GetPivot(frame) == Pivot.StaticObject) {
				Frame nextKeyFrame = Module.GetNextKey(frame); 
			    return DeriveRootRotationFrom4Joints(nextKeyFrame,  mirrored);
			 }
            else {
				RootModule module = Module.Data.GetModule(ID.Root) == null ? null : (RootModule)Module.Data.GetModule(ID.Root);
				if(module == null) {
					return Quaternion.identity;
				} else {
					return module.GetEstimatedRootRotation(reference, GetTargetTimestamp(frame, delta) - frame.Timestamp + offset , mirrored);
				}
			}
		}


		public Frame GetTargetFrame(Frame frame, float delta) {
			return Module.Data.GetFrame(GetTargetTimestamp(frame, delta));
		}

		public float GetTargetTimestamp(Frame frame, float delta) {
			if(GetPivot(frame) == Pivot.DynamicRoot || GetPivot(frame) == Pivot.DynamicObject) {
				return frame.Timestamp + GetReference(frame)/Module.Data.Framerate + delta;
			}
			if(GetPivot(frame) == Pivot.StaticRoot || GetPivot(frame) == Pivot.StaticObject) {
				return Module.Data.GetFrame(GetReference(frame)).Timestamp;
			}
			return 0f;
		}


		public void SetPivot(Frame frame, Pivot pivot) {
			if(Module.IsKey(frame) && GetPivot(frame) != pivot) {
				Pivots[frame.Index-1] = pivot;
				Compute(frame);
			}
		}

		public Pivot GetPivot(Frame frame) {
			return Pivots[frame.Index-1];
		}
		
		public void SetReference(Frame frame, int value) {
			if(Module.IsKey(frame) && GetReference(frame) != value) {
				References[frame.Index-1] = value;
				Compute(frame);
			}
		}

		public int GetReference(Frame frame) {
			return References[frame.Index-1];
		}

		public void SetProp(Frame frame, string value) {
			if(Module.IsKey(frame) && GetProp(frame) != value) {
				Props[frame.Index-1] = value;
				Compute(frame);
			}
		}

		public string GetProp(Frame frame) {
			return Props[frame.Index-1];
		}

		public void SetSpecifier(Frame frame, string value) {
			if(Module.IsKey(frame) && GetSpecifier(frame) != value) {
				Specifiers[frame.Index-1] = value;
				Compute(frame);
			}
		}

		public string GetSpecifier(Frame frame) {
			return Specifiers[frame.Index-1];
		}

		public void Compute(Frame frame) {
			Frame current = frame;
			Frame previous = Module.GetPreviousKey(current);
			Frame next = Module.GetNextKey(current);

			if(Module.IsKey(current)) {
				//Current Frame
				Pivots[current.Index-1] = GetPivot(current);
				References[current.Index-1] = GetReference(current);
				Props[current.Index-1] = GetProp(current);
				Specifiers[current.Index-1] = GetSpecifier(current);
				//Previous Frames
				if(previous != current) {
					for(int i=previous.Index; i<current.Index; i++) {
						Pivots[i-1] = GetPivot(previous);
						References[i-1] = GetReference(previous);
						Props[i-1] = GetProp(previous);
						Specifiers[i-1] = GetSpecifier(previous);
					}
				}
				//Next Frames
				if(next != current) {
					if(next.Index == Module.Data.GetTotalFrames()) {
						for(int i=current.Index+1; i<=next.Index; i++) {
							Pivots[i-1] = GetPivot(current);
							References[i-1] = GetReference(current);
							Props[i-1] = GetProp(current);
							Specifiers[i-1] = GetSpecifier(current);
						}
					} else {
						for(int i=current.Index+1; i<next.Index; i++) {
							Pivots[i-1] = GetPivot(current);
							References[i-1] = GetReference(current);
							Props[i-1] = GetProp(current);
							Specifiers[i-1] = GetSpecifier(current);
						}
					}
				}
			} else {
				for(int i=previous.Index; i<next.Index; i++) {
					Pivots[i-1] = GetPivot(previous);
					References[i-1] = GetReference(previous);
					Props[i-1] = GetProp(previous);
					Specifiers[i-1] = GetSpecifier(previous);
				}
			}
		}
	}

	[System.Serializable]
	public class GoalFunction {
		public GoalModule Module;
		public string Name;
		public float[] Values;
		public int[] TransitionCounts; // stores length of transition at every frame

		public GoalFunction(GoalModule module, string name) {
			Module = module;
			Name = name;
			Values = new float[Module.Data.GetTotalFrames()];
			TransitionCounts = new int[Module.Data.GetTotalFrames()];
		}

		public float GetValue(Frame frame) {
			return Values[frame.Index-1];
		}

		public int GetTransitionCount(Frame frame) {
			if (TransitionCounts != null && frame.Index <= (TransitionCounts.Length))
			{
                return TransitionCounts[frame.Index-1];
			} else{
				return 0;
			}
		}

		public void Toggle(Frame frame) {
			if(Module.IsKey(frame)) {
				Values[frame.Index-1] = GetValue(frame) == 1f ? 0f : 1f;
				Compute(frame);
			}
		}

		public void Compute(Frame frame) {
			Frame current = frame;
			Frame previous = Module.GetPreviousKey(current);
			Frame next = Module.GetNextKey(current);

	
			if(Module.IsKey(current)) {
				//Current Frame
				Values[current.Index-1] = GetValue(current);
				//Previous Frames
				if(previous != current) {
					for(int i=previous.Index; i<current.Index; i++) {
						Values[i-1] = GetValue(previous);
					}
				}
				//Next Frames
				if(next != current) {
					if(next.Index == Module.Data.GetTotalFrames()) {
						for(int i=current.Index+1; i<=next.Index; i++) {
							Values[i-1] = GetValue(current);
						}
					} else {
						for(int i=current.Index+1; i<next.Index; i++) {
							Values[i-1] = GetValue(current);
						}
					}
				}
			} else {
				for(int i=previous.Index; i<next.Index; i++) {
					Values[i-1] = GetValue(previous);
				}
			}
		}
	}

}
#endif