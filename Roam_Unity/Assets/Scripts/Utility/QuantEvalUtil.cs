using UnityEngine;
using UnityEngine.Rendering;
using System.IO;



#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public static class QuantEvalUtil
{	
        public static Mesh MakeReadableMeshCopy(Mesh nonReadableMesh)
        {
            Mesh meshCopy = new Mesh();
            meshCopy.indexFormat = nonReadableMesh.indexFormat;
 
            // Handle vertices
            GraphicsBuffer verticesBuffer = nonReadableMesh.GetVertexBuffer(0);
            int totalSize = verticesBuffer.stride * verticesBuffer.count;
            byte[] data = new byte[totalSize];
            verticesBuffer.GetData(data);
            meshCopy.SetVertexBufferParams(nonReadableMesh.vertexCount, nonReadableMesh.GetVertexAttributes());
            meshCopy.SetVertexBufferData(data, 0, 0, totalSize);
            verticesBuffer.Release();
 
            // Handle triangles
            meshCopy.subMeshCount = nonReadableMesh.subMeshCount;
            GraphicsBuffer indexesBuffer = nonReadableMesh.GetIndexBuffer();
            int tot = indexesBuffer.stride * indexesBuffer.count;
            byte[] indexesData = new byte[tot];
            indexesBuffer.GetData(indexesData);
            meshCopy.SetIndexBufferParams(indexesBuffer.count, nonReadableMesh.indexFormat);
            meshCopy.SetIndexBufferData(indexesData, 0, 0, tot);
            indexesBuffer.Release();
 
            // Restore submesh structure
            uint currentIndexOffset = 0;
            for (int i = 0; i < meshCopy.subMeshCount; i++)
            {
                uint subMeshIndexCount = nonReadableMesh.GetIndexCount(i);
                meshCopy.SetSubMesh(i, new SubMeshDescriptor((int)currentIndexOffset, (int)subMeshIndexCount));
                currentIndexOffset += subMeshIndexCount;
            }
 
            // Recalculate normals and bounds
            meshCopy.RecalculateNormals();
            meshCopy.RecalculateBounds();
 
            return meshCopy;
        }


        public static void SetActorPoseFromTrainingFile(Actor actor){
            string trainfile = "Assets/Demo/InitTrainPose.txt";
            float[] initData = FileUtility.ReadNthLineFromTextFile(trainfile, 1055, 0); 
            float[] currPose = initData[0..324];

            for (int j = 0; j < actor.Bones.Length; j++)
            {
             
                Vector3 bonePosition = new Vector3(currPose [j * 12], currPose [j *  12 + 1], currPose[j *  12 + 2]);
                Vector3 forward = new Vector3(currPose [j *  12 + 3], currPose [j *  12 + 4], currPose[j * 12 + 5]);
                Vector3 up = new Vector3(currPose [j *  12 + 6], currPose [j *  12 + 7], currPose[j *  12 + 8]);
                Vector3 velocity = new Vector3(currPose [j *  12 + 9], currPose [j *  12 + 10], currPose[j *  12 + 11]);
                actor.Bones[j].Transform.position = bonePosition;
                actor.Bones[j].Transform.forward = forward;
                actor.Bones[j].Transform.up = up;
                actor.Bones[j].Velocity = velocity;
            }

        }

        public static float sample_distance(){
            // return a random float between -1 and 1
            return Random.Range(2f, 4f);
        }

        public static float sample_angle(){
            // return a random float between -30 and 30
            return Random.Range(-30f, 30f);
        }


        public static void SaveToFile(string ExportPath, float[] pos_list, string filename)
        {
            if (!Directory.Exists(ExportPath))
                Directory.CreateDirectory(ExportPath);
            Debug.Log("save file to " + ExportPath + filename);
            using(TextWriter tw = new StreamWriter(ExportPath + filename))
            {
                foreach (float s in pos_list)
                    tw.WriteLine(s.ToString());
            }
        }



}