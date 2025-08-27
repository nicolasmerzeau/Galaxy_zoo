.PHONY: install

install:
	@echo ">>> Suppression TensorFlow/Keras/Numpy/Tensorboard/Protobuf"
	- pip uninstall -y tensorflow tensorflow-macos keras tensorboard protobuf numpy
	@echo ">>> Réinstallation depuis requirements.txt"
	pip install --upgrade pip setuptools wheel
	pip install --no-cache-dir -r requirements.txt
	@echo ">>> Terminé."
