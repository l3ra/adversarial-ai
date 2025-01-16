
# to update web interface
docker system prune -a --volumes  
docker build --platform linux/amd64 -t bert-inference . 
docker tag bert-inference gcr.io/adversarial-ai-445520/bert-inference
docker push gcr.io/adversarial-ai-445520/bert-inference 
kubectl delete -f chatgptdeployment.yaml
kubectl delete -f service.yaml 
kubectl apply -f chatgptdeployment.yaml
kubectl apply -f service.yaml 



