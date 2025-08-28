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
