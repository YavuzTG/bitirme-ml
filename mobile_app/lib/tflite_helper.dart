import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class TFLiteHelper {
  late Interpreter _cnnInterpreter;
  late Interpreter _lstmInterpreter;
  late TensorBuffer _cnnOutput;
  late TensorBuffer _lstmOutput;
  bool _lstmAvailable = false;
  bool _loaded = false;

  Future<void> loadModels() async {
    _cnnInterpreter = await Interpreter.fromAsset('model_cnn.tflite');
    final cnnOutShape = _cnnInterpreter.getOutputTensor(0).shape;
    final cnnOutType = _cnnInterpreter.getOutputTensor(0).type;
    _cnnOutput = TensorBuffer.createFixedSize(cnnOutShape, cnnOutType);

    try {
      _lstmInterpreter = await Interpreter.fromAsset('model_lstm.tflite');
      final lstmOutShape = _lstmInterpreter.getOutputTensor(0).shape;
      final lstmOutType = _lstmInterpreter.getOutputTensor(0).type;
      _lstmOutput = TensorBuffer.createFixedSize(lstmOutShape, lstmOutType);
      _lstmAvailable = true;
    } catch (_) {
      // LSTM model optional
      _lstmAvailable = false;
    }
    _loaded = true;
  }

  Future<Map<String, dynamic>> predictOnDevice(List<double> features) async {
    if (features.length != 16) throw ArgumentError('Expected 16 features');
    if (!_loaded) throw StateError('Models not loaded. Call loadModels() first.');

    final inputCnnShape = _cnnInterpreter.getInputTensor(0).shape;
    final inputCnnType = _cnnInterpreter.getInputTensor(0).type;
    final inputBuffer = TensorBuffer.createFixedSize(inputCnnShape, inputCnnType);
    inputBuffer.loadList(Float32List.fromList(features));

    _cnnInterpreter.run(inputBuffer.buffer, _cnnOutput.buffer);
    final cnnProbs = _cnnOutput.getDoubleList();
    final cnnIdx = cnnProbs.indexWhere((v) => v == cnnProbs.reduce((a, b) => a > b ? a : b));

    Map<String, dynamic> lstmResult = {};
    if (_lstmAvailable) {
      final inputLstmShape = _lstmInterpreter.getInputTensor(0).shape;
      final inputLstmType = _lstmInterpreter.getInputTensor(0).type;
      final inputLstmBuffer = TensorBuffer.createFixedSize(inputLstmShape, inputLstmType);
      inputLstmBuffer.loadList(Float32List.fromList(features));
      _lstmInterpreter.run(inputLstmBuffer.buffer, _lstmOutput.buffer);
      final lstmProbs = _lstmOutput.getDoubleList();
      final lstmIdx = lstmProbs.indexWhere((v) => v == lstmProbs.reduce((a, b) => a > b ? a : b));
      lstmResult = {
        'probabilities': lstmProbs,
        'predicted_class_index': lstmIdx,
      };
    }

    return {
      'cnn': {
        'probabilities': cnnProbs,
        'predicted_class_index': cnnIdx,
      },
      'lstm': lstmResult,
      // SVM: not available on-device; keep null to indicate server fallback
      'svm': null,
    };
  }

  void close() {
    try {
      _cnnInterpreter.close();
    } catch (_) {}
    try {
      if (_lstmAvailable) _lstmInterpreter.close();
    } catch (_) {}
  }
}
