#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class JointRootModule : Module {

	public enum TOPOLOGY {Biped, Quadruped, Custom};

	public TOPOLOGY Topology = TOPOLOGY.Biped;
	public int RightShoulder, LeftShoulder, RightHip, LeftHip, Neck, Hips;
	public LayerMask Ground = -1;
    public Axis ForwardAxis = Axis.ZPositive;
	public StyleModule StyleModule = null;

	public override ID GetID() {
		return ID.JointRoot;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		StyleModule = (StyleModule)data.GetModule(ID.Style);
		DetectSetup();
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
	}

	protected override void DerivedDraw(MotionEditor editor) {
		
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
		RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Data.Source.GetBoneNames());
		LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Data.Source.GetBoneNames());
		RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Data.Source.GetBoneNames());
		LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Data.Source.GetBoneNames());
		Neck = EditorGUILayout.Popup("Neck", Neck, Data.Source.GetBoneNames());
		Hips = EditorGUILayout.Popup("Hips", Hips, Data.Source.GetBoneNames());
		ForwardAxis = (Axis)EditorGUILayout.EnumPopup("Forward Axis", ForwardAxis);
		Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
	}

	public void DetectSetup() {
		MotionData.Hierarchy.Bone rs = Data.Source.FindBoneContains("RightShoulder");
		RightShoulder = rs == null ? 0 : rs.Index;
		MotionData.Hierarchy.Bone ls = Data.Source.FindBoneContains("LeftShoulder");
		LeftShoulder = ls == null ? 0 : ls.Index;
		MotionData.Hierarchy.Bone rh = Data.Source.FindBoneContains("RightUpLeg");
		RightHip = rh == null ? 0 : rh.Index;
		MotionData.Hierarchy.Bone lh = Data.Source.FindBoneContains("LeftUpLeg");
		LeftHip = lh == null ? 0 : lh.Index;
		MotionData.Hierarchy.Bone n = Data.Source.FindBoneContains("Neck");
		Neck = n == null ? 0 : n.Index;
		MotionData.Hierarchy.Bone h = Data.Source.FindBoneContains("Hips");
		Hips = h == null ? 0 : h.Index;
		Ground = LayerMask.GetMask("Ground");
	}


	public Matrix4x4 GetJointTransformation(Frame frame, bool mirrored, string boneName) {
		return Matrix4x4.TRS(GetJointPosition(frame, mirrored, boneName), GetJointRotation(frame, mirrored, boneName), Vector3.one);
	}

	public Vector3 GetJointPosition(Frame frame, bool mirrored, string boneName) {
		return frame.GetBoneTransformation(boneName, mirrored).GetPosition();
	}

	public Quaternion GetJointRotation(Frame frame, bool mirrored, string boneName) {
		return frame.GetBoneTransformation(boneName, mirrored).GetRotation();
	}

	public Vector3 GetJointVelocity(Frame frame, bool mirrored, float delta, string boneName) {
		return frame.GetBoneVelocity(boneName, mirrored, delta);
	}


	public Matrix4x4 GetEstimatedJointTransformation(Frame reference, float offset, bool mirrored, string boneName) {
		return Matrix4x4.TRS(GetEstimatedJointPosition(reference, offset, mirrored, boneName), GetEstimatedJointRotation(reference, offset, mirrored, boneName), Vector3.one);
	}

	public Vector3 GetEstimatedJointPosition(Frame reference, float offset, bool mirrored, string boneName) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return 2f*GetJointPosition(Data.GetFrame(boundary), mirrored, boneName) - GetJointPosition(Data.GetFrame(clamped), mirrored, boneName);
		} else {
			return GetJointPosition(Data.GetFrame(t), mirrored, boneName);
		}
	}


    public Vector3 GetJointGoalPositionWhenTransition(Frame reference, bool mirrored, string boneName) {
		// this function is called when walk != 1
		Frame nextKeyFrame = StyleModule.GetNextKey(reference);  
		Matrix4x4 joint = nextKeyFrame.GetBoneTransformation(boneName, mirrored);
		return joint.GetPosition();	
	}

	public Quaternion GetJointGoalRotationWhenTransition(Frame reference, bool mirrored, string boneName) {
		// this function is called when walk != 1
		Frame nextKeyFrame = StyleModule.GetNextKey(reference);  
		Matrix4x4 joint = nextKeyFrame.GetBoneTransformation(boneName, mirrored);
		return joint.GetRotation();	
	}

	public Vector3 GetPrevJointGoalPosition(Frame reference, bool mirrored, string boneName) {
		Frame prevKeyFrame = StyleModule.GetPreviousKey(reference);  
		Matrix4x4 joint = prevKeyFrame.GetBoneTransformation(boneName, mirrored);
		return joint.GetPosition();	
	}

	public Quaternion GetPrevJointGoalRotation(Frame reference, bool mirrored, string boneName) {
		Frame prevKeyFrame = StyleModule.GetPreviousKey(reference);  
		Matrix4x4 joint = prevKeyFrame.GetBoneTransformation(boneName, mirrored);
		return joint.GetRotation();	
	}

	public Quaternion GetEstimatedJointRotation(Frame reference, float offset, bool mirrored, string boneName) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return GetJointRotation(Data.GetFrame(clamped), mirrored, boneName);
		} else {
			return GetJointRotation(Data.GetFrame(t), mirrored, boneName);
		}
	}

	public Vector3 GetEstimatedJointVelocity(Frame reference, float offset, bool mirrored, float delta, string boneName) {
		return (GetEstimatedJointPosition(reference, offset + delta, mirrored, boneName) - GetEstimatedJointPosition(reference, offset, mirrored, boneName)) / delta;
	}

}
#endif
