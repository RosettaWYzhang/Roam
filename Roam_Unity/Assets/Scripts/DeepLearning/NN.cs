using UnityEngine;
using Unity.Barracuda;

namespace DeepLearning {

    [System.Serializable]
	public abstract class NN {

        public abstract class Inference {
            public int Pivot = 0;
            public double Time = 0.0;
            public abstract int GetFeedSize();

            public abstract Tensor GetFeed();

            public abstract int GetReadSize();

            public abstract void Feed(float value);
            public abstract float Read();
            public abstract void Predict();
            public bool Setup {get; set;} = false;
        }

        public Inference Session = null;

        public abstract void Create();
        public abstract void Close();

        private Inference GetSession() {
            if(Session == null) {
                Debug.Log("No session found.");
            }
            return Session;
        }

        public void CreateSession() {
            if(Session != null) {
                Debug.Log("Session is already active.");
            } else {
                Create();
            }
        }

        public void CloseSession() {
            if(Session == null) { 
                Debug.Log("No session currently active.");
            } else {
                Close();
                Session = null;
            }
        }

        public void Predict() {
            if(GetSession() != null) {
                if(Session.Pivot != Session.GetFeedSize()) {
                    Debug.Log("Running prediction without all inputs given to the network: " + Session.Pivot + " / " + Session.GetFeedSize());
                }
                System.DateTime timestamp = Utility.GetTimestamp();
                Session.Predict();
                Session.Time = Utility.GetElapsedTime(timestamp);
                Session.Pivot = Session.GetReadSize();
            }
        }
        

        public void SetPivot(int index) {
            Session.Pivot = index;
        }

        public int GetPivot() {
            return Session.Pivot;
        }

        public void ResetPivot() {
            Session.Pivot = 0;
        }
        
		public void Feed(float value) {
            if(GetSession() != null) {
                if(Session.Pivot == Session.GetFeedSize()) {
                    Debug.Log(Session.GetFeedSize());

                    Debug.Log("Attempting to feed more values than inputs available.");
                } else {
                    Session.Feed(value);
                    Session.Pivot += 1;

                }
            }
		}

        public void Feed(bool value) {
            Feed(value ? 1f : 0f);
        }

        public void Feed(float[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(bool[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(Vector2 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void Feed(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.y);
            Feed(vector.z);
        }

        public void FeedXY(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void FeedXZ(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.z);
        }

        public void FeedYZ(Vector3 vector) {
            Feed(vector.y);
            Feed(vector.z);
        }


		public float Read() {
            float value = 0f;
            if(GetSession() != null) {
                if(Session.Pivot == Session.GetReadSize()) {
                    Debug.Log("Attempting to read more values than outputs available.");
                } else {
                    value = Session.Read();
                    Session.Pivot += 1;
                }
            }
            return value;
		}

    	public float Read(float min, float max) {
            return Mathf.Clamp(Read(), min, max);
		}

        public float[] Read(int count) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read();
            }
            return values;
        }

        public float[] Read(int count, float min, float max) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read(min, max);
            }
            return values;
        }

        public Vector3 ReadVector2() {
            return new Vector2(Read(), Read());
        }

        public Vector3 ReadVector3() {
            return new Vector3(Read(), Read(), Read());
        }

        public Vector3 ReadXY() {
            return new Vector3(Read(), Read(), 0f);
        }

        public Vector3 ReadXZ() {
            return new Vector3(Read(), 0f, Read());
        }

        public Vector3 ReadYZ() {
            return new Vector3(0f, Read(), Read());
        }

    }

}