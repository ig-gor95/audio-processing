!pip install pyannote.audio
!pip install openai-whisper
!pip install torchaudio
!pip install -q torch
!pip install python-Levenshtein
!pip install pydu
!pip install speechbrain
!pip install torchaudio
!pip install silero-vad
!pip install tqdm
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension
!jupyter labextension install @jupyter-widgets/jupyterlab-manager
# !pip install labextension
#!conda install -c conda-forge ipywidgets может понадобиться




audio_analysis_project/  
├── configs/                       # Конфигурации под каждого клиента  
│   ├── client_A.yaml              # Критерии для заказчика A  
│   └── client_B.yaml              # Критерии для заказчика B  
├── core/                          # Ядро системы  
│   ├── audio_loader.py            # Загрузка и предобработка аудио  
│   ├── speech_to_text.py          # ASR (Vosk/Whisper)  
│   ├── analyzers/                 # Анализаторы (каждый критерий — отдельный файл)  
│   │   ├── pause_analyzer.py      # Расчет пауз  
│   │   ├── swear_detector.py      # Поиск матов  
│   │   └── script_checker.py      # Проверка скриптов  
│   └── pipeline.py                # Главный пайплайн  
├── data/                          # Данные  
│   ├── input/                     # Исходные аудиофайлы  
│   └── output/                    # Отчеты (Excel/PDF)  
├── models/                        # ML-модели (если используются)  
│   ├── swear_classifier.pkl       # Модель для матов  
│   └── tone_analysis.h5           # Анализ тональности  
├── tests/                         # Юнит-тесты  
├── utils/                         # Вспомогательные скрипты  
│   ├── logger.py                  # Логирование  
│   └── config_loader.py           # Загрузка YAML-конфигов  
└── app.py                         # FastAPI/REST-интерфейс (опционально)  
