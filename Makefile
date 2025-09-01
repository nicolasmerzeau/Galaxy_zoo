.PHONY: install

reinstall_package:
	@pip uninstall -y Galaxy_zoo || :
	@pip install -e .


install:
	@echo ">>> Suppression TensorFlow/Keras/Numpy/Tensorboard/Protobuf"
	- pip uninstall -y tensorflow tensorflow-macos keras tensorboard protobuf numpy
	@echo ">>> Réinstallation depuis requirements.txt"
	pip install --upgrade pip setuptools wheel
	pip install --no-cache-dir -r requirements.txt
	@echo ">>> Terminé."

run_all_models:
	python -c 'from galaxy_zoo.models.compare_models import run_models; run_models()'

run_all_models_light:
	python -c 'from galaxy_zoo.models.compare_models_light import run_models; run_models()'

deploy:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/galaxy-zoo/${GAR_IMAGE}:prod .
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/galaxy-zoo/${GAR_IMAGE}:prod
	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/galaxy-zoo/${GAR_IMAGE}:prod --memory ${GAR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml
