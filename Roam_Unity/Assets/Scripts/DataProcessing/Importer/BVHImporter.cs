﻿#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class BVHImporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public bool Flip = true;
    public bool IgnoreRootPosition = false;
	public Axis Axis = Axis.XPositive;

	public string Source = "";
	public string Destination = "MotionCapture";
	public string Filter = string.Empty;
	public File[] Files = new File[0];
	public File[] Instances = new File[0];
	public bool Importing = false;
	
	public int Page = 1;
	public const int Items = 25;

	[MenuItem ("AI4Animation/BVH Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHImporter));
		Scroll = Vector3.zero;
	}
	
	void OnGUI() {
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("BVH Importer");
				}
		
				if(!Importing) {
					if(Utility.GUIButton("Import Motion Data", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ImportMotionData());
					}
				} else {
					if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
						this.StopAllCoroutines();
						Importing = false;
					}
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Source");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					Source = EditorGUILayout.TextField(Source);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Source = EditorUtility.OpenFolderPanel("BVH Importer", Source == string.Empty ? Application.dataPath : Source, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.LabelField("Destination");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
					Destination = EditorGUILayout.TextField(Destination);
					EditorGUILayout.EndHorizontal();

					string filter = EditorGUILayout.TextField("Filter", Filter);
					if(Filter != filter) {
						Filter = filter;
						ApplyFilter();
					}

					Flip = EditorGUILayout.Toggle("Flip", Flip);
                    IgnoreRootPosition = EditorGUILayout.Toggle("IgnoreRootPosition", IgnoreRootPosition);
                    Axis = (Axis)EditorGUILayout.EnumPopup("Axis", Axis);

					if(Utility.GUIButton("Load Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
						LoadDirectory();
					}

					int start = (Page-1)*Items;
					int end = Mathf.Min(start+Items, Instances.Length);
					int pages = Mathf.CeilToInt(Instances.Length/Items)+1;
					Utility.SetGUIColor(UltiDraw.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Max(Page-1, 1);
						}
						EditorGUILayout.LabelField("Page " + Page + "/" + pages);
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Min(Page+1, pages);
						}
						EditorGUILayout.EndHorizontal();
					}
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Import = true;
						}
					}
					if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Import = false;
						}
					}
					EditorGUILayout.EndHorizontal();
					for(int i=start; i<end; i++) {
						if(Instances[i].Import) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Import = EditorGUILayout.Toggle(Instances[i].Import, GUILayout.Width(20f));
							EditorGUILayout.LabelField(Instances[i].Object.Name);
							EditorGUILayout.EndHorizontal();
						}
					}
				}
		
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory() {
		if(Directory.Exists(Source)) {
			DirectoryInfo info = new DirectoryInfo(Source);
			FileInfo[] items = info.GetFiles("*.bvh");
			Files = new File[items.Length];
			for(int i=0; i<items.Length; i++) {
				Files[i] = new File();
				Files[i].Object = items[i];
				Files[i].Import = true;
			}
		} else {
			Files = new File[0];
		}
		ApplyFilter();
		Page = 1;
	}

	private void ApplyFilter() {
		if(Filter == string.Empty) {
			Instances = Files;
		} else {
			List<File> instances = new List<File>();
			for(int i=0; i<Files.Length; i++) {
				if(Files[i].Object.Name.ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Files[i]);
				}
			}
			Instances = instances.ToArray();
		}
	}

	private IEnumerator ImportMotionData() {
		string destination = "Assets/" + Destination;
		if(!AssetDatabase.IsValidFolder(destination)) {
			Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
		} else {
			Importing = true;
			for(int f=0; f<Files.Length; f++) {
				if(Files[f].Import) {
					string fileName = Files[f].Object.Name.Replace(".bvh", "");
					if(!Directory.Exists(destination+"/"+fileName) ) {
						AssetDatabase.CreateFolder(destination, fileName);
						MotionData data = ScriptableObject.CreateInstance<MotionData>();
						data.name = fileName;
						AssetDatabase.CreateAsset(data, destination+"/"+fileName+"/"+data.name+".asset");

						string[] lines = System.IO.File.ReadAllLines(Files[f].Object.FullName);
						char[] whitespace = new char[] {' ','\t' };
						int index = 0;

						//Create Source Data
						List<Vector3> offsets = new List<Vector3>();
						List<int[]> channels = new List<int[]>();
						List<float[]> motions = new List<float[]>();
						data.Source = new MotionData.Hierarchy();
						string name = string.Empty;
						string parent = string.Empty;
						Vector3 offset = Vector3.zero;
						int[] channel = null;
						for(index = 0; index<lines.Length; index++) {
							if(lines[index] == "MOTION") {
								break;
							}
							string[] entries = lines[index].Split(whitespace);
							for(int entry=0; entry<entries.Length; entry++) {
								if(entries[entry].Contains("ROOT")) {
									parent = "None";
									name = entries[entry+1];
									break;
								} else if(entries[entry].Contains("JOINT")) {
									parent = name;
									name = entries[entry+1];
									break;
								} else if(entries[entry].Contains("End")) {
									parent = name;
									name = name+entries[entry+1];
									string[] subEntries = lines[index+2].Split(whitespace);									
									for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {										
										if(subEntries[subEntry].Contains("OFFSET")) {
											offset.x = FileUtility.ReadFloat(subEntries[subEntry+1]);
											offset.y = FileUtility.ReadFloat(subEntries[subEntry+2]);
											offset.z = FileUtility.ReadFloat(subEntries[subEntry+3]);
											break;
										}
									}
									data.Source.AddBone(name, parent);
									offsets.Add(offset);
									channels.Add(new int[0]);
									index += 2;
									break;
								} else if(entries[entry].Contains("OFFSET")) {
									offset.x = FileUtility.ReadFloat(entries[entry+1]);
									offset.y = FileUtility.ReadFloat(entries[entry+2]);
									offset.z = FileUtility.ReadFloat(entries[entry+3]);
									break;
								} else if(entries[entry].Contains("CHANNELS")) {
									channel = new int[FileUtility.ReadInt(entries[entry+1])];
									for(int i=0; i<channel.Length; i++) {
										if(entries[entry+2+i] == "Xposition") {
											channel[i] = 1;
										} else if(entries[entry+2+i] == "Yposition") {
											channel[i] = 2;
										} else if(entries[entry+2+i] == "Zposition") {
											channel[i] = 3;
										} else if(entries[entry+2+i] == "Xrotation") {
											channel[i] = 4;
										} else if(entries[entry+2+i] == "Yrotation") {
											channel[i] = 5;
										} else if(entries[entry+2+i] == "Zrotation") {
											channel[i] = 6;
										}
									}
									data.Source.AddBone(name, parent);
									offsets.Add(offset);
									channels.Add(channel);
									break;
								} else if(entries[entry].Contains("}")) {
									name = parent;
									parent = name == "None" ? "None" : data.Source.FindBone(name).Parent;
									break;
								}
							}
						}

						//Set Frames
						index += 1;
						while(lines[index].Length == 0) {
							index += 1;
						}
						ArrayExtensions.Resize(ref data.Frames, FileUtility.ReadInt(lines[index].Substring(8)));

						//Set Framerate
						index += 1;
						data.Framerate = Mathf.RoundToInt(1f / FileUtility.ReadFloat(lines[index].Substring(12)));
						//Compute Frames
						index += 1;
						for(int i=index; i<lines.Length; i++) {
							motions.Add(FileUtility.ReadArray(lines[i]));
						}
						for(int k=0; k<data.GetTotalFrames(); k++) {
							data.Frames[k] = new Frame(data, k+1, (float)k / data.Framerate);
							int idx = 0;
							for(int i=0; i<data.Source.Bones.Length; i++) {
								MotionData.Hierarchy.Bone info = data.Source.Bones[i];
								Vector3 position = Vector3.zero;
								Quaternion rotation = Quaternion.identity;
                                
								for(int j=0; j<channels[i].Length; j++) {

									if(channels[i][j] == 1) {
										position.x = motions[k][idx]; idx += 1;
									}
									if(channels[i][j] == 2) {
										position.y = motions[k][idx]; idx += 1;
									}
									if(channels[i][j] == 3) {
										position.z = motions[k][idx]; idx += 1;
									}
									// Rotate whatever comes first
                                    // X is right
                                    if (channels[i][j] == 4) {
										rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.right); idx += 1;
									}
									// Y is up
									if(channels[i][j] == 5) {
										rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.up); idx += 1;
									}
									// Z is forward
									if(channels[i][j] == 6) {
										rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.forward); idx += 1;
									}
								}
                                if (i == 0 && IgnoreRootPosition)
                                {
                                    position = Vector3.zero;
                                }

                                position = (position == Vector3.zero ? offsets[i] : position); //unit scale
								// position: swap y and z axis
                                Matrix4x4 local = Matrix4x4.TRS(position, rotation, Vector3.one);
								if (info.Parent == "None"){
								    Matrix4x4 permute = new Matrix4x4(new Vector4(1, 0, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 0, 1));
									local = permute * local;
								}

								if(Flip) {
									local = local.GetMirror(Axis);
								}
								data.Frames[k].World[i] = info.Parent == "None" ? local : data.Frames[k].World[data.Source.FindBone(info.Parent).Index] * local;
							}

						}

						if(data.GetTotalFrames() == 1) {
							Frame reference = data.Frames.First();
							ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt(data.Framerate));
							for(int k=0; k<data.GetTotalFrames(); k++) {
								data.Frames[k] = new Frame(data, k+1, (float)k / data.Framerate);
								data.Frames[k].World = (Matrix4x4[])reference.World.Clone();
							}
						}

						//Detect Symmetry
						data.DetectSymmetry();

						//Add Scene
						data.CreateScene();
						data.AddSequence();

						//Save
						EditorUtility.SetDirty(data);
					} else {
						Debug.Log("File with name " + fileName + " already exists.");
					}

					yield return new WaitForSeconds(0f);
				}
			}
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
			Importing = false;
		}
		yield return new WaitForSeconds(0f);
	}

	[System.Serializable]
	public class File {
		public FileInfo Object;
		public bool Import;
	}

}
#endif
