import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'tflite_helper.dart';

void main() {
  runApp(const BeedMobileApp());
}

class BeedMobileApp extends StatelessWidget {
  const BeedMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'BEED Mobil',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF14B8A6)),
        scaffoldBackgroundColor: const Color(0xFF07111F),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
    String _baseUrl = 'http://10.0.2.2:8000';
    late final TextEditingController _baseUrlController;

    final List<TextEditingController> _controllers =
      List.generate(16, (_) => TextEditingController(text: '0'));

  bool _loading = false;
  String _result = 'Tahmin sonucu burada görünecek.';
  final TFLiteHelper _tflite = TFLiteHelper();
  bool _tfliteAvailable = false;

  @override
  void dispose() {
    _baseUrlController.dispose();
    for (final controller in _controllers) {
      controller.dispose();
    }
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController(text: _baseUrl);
    // Try to load TFLite models; if assets missing, keep server fallback
    _initTflite();
  }

  Future<void> _initTflite() async {
    try {
      await _tflite.loadModels();
      setState(() {
        _tfliteAvailable = true;
      });
    } catch (e) {
      // assets not present or load failed — keep server fallback
      setState(() {
        _tfliteAvailable = false;
      });
    }
  }

  void _resetFields() {
    setState(() {
      for (final controller in _controllers) {
        controller.text = '0';
      }
      _result = 'Tahmin sonucu burada görünecek.';
    });
  }

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _result = 'Tahmin hesaplanıyor...';
    });

    try {
      final features = _controllers
          .map((controller) => double.tryParse(controller.text.trim()) ?? 0.0)
          .toList();
      if (_tfliteAvailable) {
        final res = await _tflite.predictOnDevice(features);
        final cnn = res['cnn'] as Map<String, dynamic>;
        final lstm = res['lstm'] as Map<String, dynamic>;
        setState(() {
          _result = [
            'CNN: class ${cnn['predicted_class_index']} / confidence=${cnn['probabilities'][cnn['predicted_class_index']]}',
            lstm.isNotEmpty
                ? 'LSTM: class ${lstm['predicted_class_index']} / confidence=${lstm['probabilities'][lstm['predicted_class_index']]}'
                : 'LSTM: not available',
            'SVM: server fallback',
          ].join('\n');
        });
      } else {
        final response = await http.post(
          Uri.parse('$_baseUrl/predict'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({'features': features}),
        );

        final decoded = jsonDecode(response.body);
        if (response.statusCode >= 400) {
          final message = decoded is Map<String, dynamic>
              ? decoded['detail']?.toString() ?? 'Tahmin alınamadı.'
              : 'Tahmin alınamadı.';
          throw Exception(message);
        }

        final predictions = decoded['predictions'] as Map<String, dynamic>;
        final cnn = predictions['cnn'] as Map<String, dynamic>;
        final svm = predictions['svm'] as Map<String, dynamic>;
        final lstm = predictions['lstm'] as Map<String, dynamic>;

        setState(() {
          _result = [
            'CNN: class ${cnn['predicted_class_index']} / y=${cnn['predicted_y_label']} / confidence=${cnn['confidence']}',
            'SVM: class ${svm['predicted_class_index']} / y=${svm['predicted_y_label']}',
            'LSTM: class ${lstm['predicted_class_index']} / y=${lstm['predicted_y_label']} / confidence=${lstm['confidence']}',
          ].join('\n');
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Hata: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF07111F), Color(0xFF0B1730), Color(0xFF06131B)],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 720),
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(18),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const SizedBox(height: 8),
                    const Text(
                      'BEED Mobil Tahmin',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 30,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '16 özelliği gir ve backend modeli emülatörden çağır.',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 18),
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: const Color(0xFF101B2D).withOpacity(0.92),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(color: Colors.white.withOpacity(0.08)),
                        boxShadow: const [
                          BoxShadow(
                            color: Colors.black45,
                            blurRadius: 30,
                            offset: Offset(0, 18),
                          ),
                        ],
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          // Backend URL input for real device usage
                          Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _baseUrlController,
                                  style: const TextStyle(color: Colors.white),
                                  decoration: InputDecoration(
                                    filled: true,
                                    fillColor: Colors.white.withOpacity(0.06),
                                    hintText: 'Backend URL (e.g. http://192.168.1.5:8000)',
                                    hintStyle: TextStyle(color: Colors.white.withOpacity(0.4)),
                                    border: OutlineInputBorder(
                                      borderRadius: BorderRadius.circular(12),
                                      borderSide: BorderSide.none,
                                    ),
                                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              IconButton(
                                onPressed: () => setState(() {
                                  _baseUrl = _baseUrlController.text.trim();
                                }),
                                icon: const Icon(Icons.check, color: Colors.white),
                                tooltip: 'Apply backend URL',
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          GridView.builder(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            itemCount: _controllers.length,
                            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 2,
                              mainAxisSpacing: 12,
                              crossAxisSpacing: 12,
                              childAspectRatio: 1.7,
                            ),
                            itemBuilder: (context, index) {
                              return _FeatureInput(
                                label: 'X${index + 1}',
                                controller: _controllers[index],
                              );
                            },
                          ),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Expanded(
                                child: FilledButton(
                                  onPressed: _loading ? null : _predict,
                                  style: FilledButton.styleFrom(
                                    padding: const EdgeInsets.symmetric(vertical: 14),
                                    backgroundColor: const Color(0xFF14B8A6),
                                    foregroundColor: Colors.white,
                                  ),
                                  child: _loading
                                      ? const SizedBox(
                                          height: 18,
                                          width: 18,
                                          child: CircularProgressIndicator(
                                            strokeWidth: 2,
                                            color: Colors.white,
                                          ),
                                        )
                                      : const Text('Tahmin Et'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              OutlinedButton(
                                onPressed: _loading ? null : _resetFields,
                                style: OutlinedButton.styleFrom(
                                  padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 18),
                                  foregroundColor: Colors.white,
                                  side: BorderSide(color: Colors.white.withOpacity(0.15)),
                                ),
                                child: const Text('Sıfırla'),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.96),
                              borderRadius: BorderRadius.circular(18),
                            ),
                            child: Text(
                              _result,
                              style: const TextStyle(
                                color: Color(0xFF0F172A),
                                fontSize: 15,
                                height: 1.45,
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Backend adresi: $_baseUrl',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.6),
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _FeatureInput extends StatelessWidget {
  const _FeatureInput({required this.label, required this.controller});

  final String label;
  final TextEditingController controller;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withOpacity(0.7),
            fontSize: 12,
            letterSpacing: 1.1,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        TextField(
          controller: controller,
          keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            filled: true,
            fillColor: Colors.white.withOpacity(0.07),
            hintText: '0',
            hintStyle: TextStyle(color: Colors.white.withOpacity(0.35)),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(16),
              borderSide: BorderSide.none,
            ),
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
          ),
        ),
      ],
    );
  }
}
