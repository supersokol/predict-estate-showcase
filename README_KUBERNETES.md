### README: How to Launch the PredictEstate System via Kubernetes

This guide explains how to set up and launch the **PredictEstate** system using Docker containers and Kubernetes.

### **Step 1: Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone https://github.com/supersokol/predict-estate-showcase.git
cd predict-estate-showcase
```

### **Step 2: Build Docker Containers**
Build the Docker containers for all services:
```bash
docker-compose build
```

### **Step 3: Push Docker Images to a Registry**
Tag and push your Docker images to your container registry:
```bash
docker tag predictestate-api:latest <your_registry>/predictestate-api:latest
docker push <your_registry>/predictestate-api:latest
```
Repeat this process for all other containers (`streamlit`, `logstash`, `airflow`, etc.).

### **Step 4: Deploy the System Using Kubernetes**
Apply Kubernetes manifests to launch the services in a Kubernetes cluster:
```bash
kubectl apply -f k8s/
```

Check the status of your pods to ensure everything is running:
```bash
kubectl get pods
```

### **Step 5: Access the System**
- Open **Streamlit** for testing:
  ```bash
  http://predictestate.local/streamlit
  ```

### **Additional Notes**
- **Ingress Configuration**: Ensure your `Ingress` is correctly set up and your DNS resolves to the Kubernetes cluster.
- **Dependencies**: Ensure Kubernetes is installed and configured, and Docker images are accessible from your cluster.
=
Enjoy using PredictEstate!