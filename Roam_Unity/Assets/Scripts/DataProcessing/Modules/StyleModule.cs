﻿#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using System.IO;

public class StyleModule : Module {

	public bool[] Keys = new bool[0];
	public StyleFunction[] Functions = new StyleFunction[0];

	public override ID GetID() {
		return ID.Style;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Functions = new StyleFunction[0];
		Keys = new bool[data.GetTotalFrames()];
		Keys[0] = true;
		Keys[Keys.Length-1] = true;
		return this;
	}

	public override void Slice(Sequence sequence) {
		Keys = ArrayExtensions.Gather(ref Keys, sequence.GetIndices());
		for(int i=0; i<Functions.Length; i++) {
			Functions[i].Values = ArrayExtensions.Gather(ref Functions[i].Values, sequence.GetIndices());
		}
	}

	public override void Callback(MotionEditor editor) {
		
	}

	public void AddStyle(string name) {
		if(System.Array.Exists(Functions, x => x.Name == name)) {
			Debug.Log("Style with name " + name + " already exists.");
			return;
		}
		ArrayExtensions.Add(ref Functions, new StyleFunction(this, name));
	}

	public void RemoveStyle(string name) {
		int index = System.Array.FindIndex(Functions, x => x.Name == name);
		if(index >= 0) {
			ArrayExtensions.RemoveAt(ref Functions, index);
		} else {
			Debug.Log("Style with name " + name + " does not exist.");
		}
	}

	public StyleFunction GetStyleFunction(string name) {
		return System.Array.Find(Functions, x => x.Name == name);
	}

	public float[] GetStyles(Frame frame) {
		float[] style = new float[Functions.Length];
		for(int i=0; i<style.Length; i++) {
			style[i] = Functions[i].GetValue(frame);
		}
		return style;
	}

	public float[] GetStyles(Frame frame, params string[] names) {
		float[] style = new float[names.Length];
		for(int i=0; i<style.Length; i++) {
			style[i] = GetStyle(frame, names[i]);
		}
		return style;
	}

	public float GetStyle(Frame frame, string name) {
		StyleFunction function = GetStyleFunction(name);
		return function == null ? 0f : function.GetValue(frame);
	}

	public float GetStyle(Frame frame, int index) {
		return Functions[index].GetValue(frame);
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
		for(int i=0; i<Functions.Length; i++) {
			Functions[i].Compute(frame);
		}
	}

	public bool IsKey(Frame frame) {
		return Keys[frame.Index-1];
	}
	public bool IsKeyNoMinus(Frame frame) {
		return Keys[frame.Index];
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

		EditorGUI.BeginDisabledGroup(!IsKey(frame));
		Color[] colors = UltiDraw.GetRainbowColors(Functions.Length);
		for(int i=0; i<Functions.Length; i++) {
			float height = 25f;
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton(Functions[i].Name, colors[i].Transparent(Utility.Normalise(Functions[i].GetValue(frame), 0f, 1f, 0.25f, 1f)), UltiDraw.White, 150f, height)) {
				Functions[i].Toggle(frame);
			}

			Rect c = EditorGUILayout.GetControlRect();
			Rect r = new Rect(c.x, c.y, Functions[i].GetValue(frame) * c.width, height);
			EditorGUI.DrawRect(r, colors[i].Transparent(0.75f));
			Functions[i].SetValue(frame, EditorGUILayout.FloatField(Functions[i].GetValue(frame), GUILayout.Width(50f)));
			Functions[i].Name = EditorGUILayout.TextField(Functions[i].Name);
			if(Utility.GUIButton("Smooth", UltiDraw.DarkGrey, UltiDraw.White, 80f, 20f)) {
				Functions[i].Smooth(frame);
			}
			if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f)) {
				RemoveStyle(Functions[i].Name);
			}
			EditorGUILayout.EndHorizontal();
		}
		EditorGUI.EndDisabledGroup();
		if (Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
			AddStyle("Style " + (Functions.Length+1));
			EditorGUIUtility.ExitGUI();
		}
		if (Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White))
		{
			Clear();
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


		//Styles
		for(int i=0; i<Functions.Length; i++) {
			Frame current = Data.Frames.First();
			while(current != Data.Frames.Last()) {
				Frame next = GetNextKey(current);
				float _start = (float)(Mathf.Clamp(current.Index, start, end)-start) / (float)elements;
				float _end = (float)(Mathf.Clamp(next.Index, start, end)-start) / (float)elements;
				float xStart = rect.x + _start * rect.width;
				float xEnd = rect.x + _end * rect.width;
				float yStart = rect.y + (1f - Functions[i].Values[Mathf.Clamp(current.Index, start, end)-1]) * rect.height;
				float yEnd = rect.y + (1f - Functions[i].Values[Mathf.Clamp(next.Index, start, end)-1]) * rect.height;
				UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
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

    private void Clear()
    {
        for (int f = 0; f < Functions.Length; f++)
		{
			Functions[f].Values = new float[Data.GetTotalFrames()]; 
		}
		Keys = new bool[Data.GetTotalFrames()];
		Keys[0] = true;
		Keys[Keys.Length - 1] = true;
    }

	public float[] SimulateTimestamps(Frame frame, int padding) {
		float step = 1f/25f; // framerate
		float start = frame.Timestamp - padding*step;
		float[] timestamps = new float[2*padding+1];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = start + i*step;
		}
		return timestamps;
	}
	public void GenerateKeys() {
		for(int i=0; i<Keys.Length; i++) {
			Keys[i] = false;
		}
		Keys[0] = true;
		Keys[Keys.Length-1] = true;
		foreach(StyleFunction f in Functions) {
			for(int i=1; i<f.Values.Length-1; i++) {
				if((f.Values[i] == 0f || f.Values[i] == 1f) && (f.Values[i] != f.Values[i-1] || f.Values[i] != f.Values[i+1])) {
					Keys[i] = true;
				}
			}
		}
	}

	

    [System.Serializable]
	public class StyleFunction {
		public StyleModule Module;
		public string Name;
		public float[] Values;

		public StyleFunction(StyleModule module, string name) {
			Module = module;
			Name = name;
			Values = new float[Module.Data.GetTotalFrames()];
		}


		public void SetValue(Frame frame, float value) {
			if(Values[frame.Index-1] != value) {
				Values[frame.Index-1] = value;
				Compute(frame);
			}
		}

		public float GetValue(Frame frame) {
			return Values[frame.Index-1];
		}

		public void Smooth(Frame frame) {
			Frame previous = Module.GetPreviousKey(frame);
			Frame next = Module.GetNextKey(frame);
			float weight = (frame.Timestamp - previous.Timestamp) / (next.Timestamp - previous.Timestamp);
			float value = (1f-weight) * GetValue(previous) + weight * GetValue(next);
			SetValue(frame, value);
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

			if(Module.IsKey(frame)) {
				//Current Frame
				Values[current.Index-1] = GetValue(current);
				//Previous Frames
				if(previous != frame) {
					float valA = GetValue(previous);
					float valB = GetValue(current);
					for(int i=previous.Index; i<current.Index; i++) {
						float weight = (float)(i-previous.Index) / (float)(frame.Index - previous.Index);
						Values[i-1] = (1f-weight) * valA + weight * valB;
					}
				}
				//Next Frames
				if(next != frame) {
					float valA = GetValue(current);
					float valB = GetValue(next);
					for(int i=current.Index+1; i<=next.Index; i++) {
						float weight = (float)(i-current.Index) / (float)(next.Index - current.Index);
						Values[i-1] = (1f-weight) * valA + weight * valB;
					}
				}
			} else {
				float valA = GetValue(previous);
				float valB = GetValue(next);
				for(int i=previous.Index; i<=next.Index; i++) {
					float weight = (float)(i-previous.Index) / (float)(next.Index - previous.Index);
					Values[i-1] = (1f-weight) * valA + weight * valB;
				}
			}
		}
	}

}
#endif