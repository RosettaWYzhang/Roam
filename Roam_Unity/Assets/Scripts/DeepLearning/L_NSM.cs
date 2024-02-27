using Unity.Barracuda;
using System;
using System.IO;
using System.Linq;

namespace DeepLearning
{
    [System.Serializable]
    public class L_NSM : NeuralNetwork
    {

        public NNModel modelAsset;
        private Model m_RuntimeModel;
        private IWorker worker;

        private int x_dim = 1055;
        private int output_dim = 871;

        private Tensor x;

        private float[] y;

        private bool verbose = false;

        int[,] Intervals;


        protected override bool SetupDerived()
        {
            if (Setup)
            {
                return true;
            }
            LoadDerived();
            Setup = true;
            return true;
        }

        protected override bool ShutdownDerived()
        {
            if (Setup)
            {
                UnloadDerived();
                ResetPredictionTime();
                ResetPivot();
            }
            return false;
        }

        protected void LoadDerived()
        {
            m_RuntimeModel = ModelLoader.Load(modelAsset, verbose);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RuntimeModel, verbose);

            x = new Tensor(1, x_dim);
            y = new float[output_dim];
            Intervals = new[,] { { 0, x_dim } };
        }

        protected void UnloadDerived()
        {

        }

        public void OnDestroy()
        {
            worker?.Dispose();
            x.Dispose();
        }

        public void Normalize(ref Tensor X, Tensor Xmean, Tensor Xstd)
        {
            for (int i = 0; i < X.length; i++)
            {
                X[i] = (X[i] - Xmean[i]) / Xstd[i];
            }
        }

        public void UnNormalize(ref Tensor X, Tensor Xmean, Tensor Xstd)
        {
            for (int i = 0; i < X.length; i++)
            {
                X[i] = X[i] * Xstd[i] + Xmean[i];
            }
        }

        protected override void PredictDerived()
        {
            worker.Execute(x);
            Tensor output = worker.PeekOutput();
            for (int i = 0; i < output.length; i++)
            {
                y[i] = output[i];

            }
        }

        public override void SetInput(int index, float value)
        {

            if (Setup)
            {

                if (index >= Intervals[0, 0] && index < Intervals[0, 1])
                {
                    x[0, index - Intervals[0, 0]] = value;
                }

                else
                {
                    throw new System.InvalidOperationException("Insertion exceded the allocated input size");
                }


            }
        }

        public override float GetOutput(int index)
        {
            if (Setup)
            {
                return y[index];
            }
            else
            {
                return 0f;
            }
        }

    }


}

