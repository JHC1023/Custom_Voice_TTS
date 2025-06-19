import csv
import os
import time
import sounddevice as sd
import speech_recognition as sr
import keyboard
import Levenshtein
import numpy as np
from datetime import datetime
import re
import wave
import scipy.io.wavfile

# 파일 경로 설정
CSV_FILE = "korean_corpus.csv"  # 문장이 저장된 CSV 파일
PROGRESS_FILE = "recording_progress.txt"  # 진행 상황 저장 파일
OUTPUT_DIR = "recordings"  # 녹음 파일 저장 폴더

# 출력 폴더가 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 녹음 설정
SAMPLE_RATE = 44100  # 샘플레이트 (Hz)
RECORD_KEY = 'space'  # 녹음 시작/종료 키

def load_sentences():
    """CSV 파일에서 한국어 문장을 불러옴"""
    sentences = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # 행이 비어있지 않은지 확인
                # 문장 끝의 숫자 제거 (예: "안녕하세요 123" -> "안녕하세요")
                sentence = re.sub(r'\s+\d+$', '', row[0].strip())
                sentences.append(sentence)
    return sentences

def load_progress():
    """마지막으로 녹음한 문장 인덱스를 진행 파일에서 불러옴"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def save_progress(index):
    """현재 문장 인덱스를 진행 파일에 저장"""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def record_audio(filename):
    """스페이스바를 한 번 누르면 녹음 시작, 다시 누르면 종료"""
    print(f"'{RECORD_KEY}' 키를 눌러 녹음을 시작하세요...")
    keyboard.wait(RECORD_KEY)  # 첫 번째 누름으로 녹음 시작
    print("녹음 시작...")
    recording = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1)
    stream.start()
    try:
        while True:
            data, overflowed = stream.read(1024)
            if overflowed:
                print("오버플로우 발생: 데이터 손실 가능성")
            recording.append(data)
            # 스페이스바 누름 이벤트 감지
            if keyboard.is_pressed(RECORD_KEY):
                break
    finally:
        stream.stop()
        stream.close()  # 스트림 명시적 종료
    print("녹음 종료.")
    # 녹음 데이터 확인
    if not recording:
        print("오류: 녹음 데이터가 비어 있습니다. 다시 녹음하세요.")
        return None
    # 리스트를 numpy 배열로 변환
    recording = np.concatenate(recording, axis=0)
    # 부동소수점 데이터를 16비트 정수로 변환
    recording = (recording * 32767).astype(np.int16)
    # WAV 파일로 저장 (wave 모듈 사용)
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(2)  # 16비트
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording.tobytes())
        print(f"녹음 파일 저장됨: {filename}")
        # 파일 존재 여부 확인
        if os.path.exists(filename):
            print(f"파일 확인됨: {os.path.getsize(filename)} 바이트")
        else:
            print(f"오류: 파일이 생성되지 않음: {filename}")
        return filename
    except Exception as e:
        print(f"파일 저장 오류: {e}")
        return None

def speech_to_text(audio_file):
    """녹음된 오디오를 텍스트로 변환"""
    if not audio_file or not os.path.exists(audio_file):
        print("오류: 오디오 파일이 존재하지 않습니다.")
        return ""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        # 한국어로 Google 음성 인식 사용
        text = recognizer.recognize_google(audio, language='ko-KR')
        return text
    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
        return ""
    except sr.RequestError as e:
        print(f"음성 인식 오류: {e}")
        return ""
    except Exception as e:
        print(f"오디오 파일 처리 오류: {e}")
        return ""

def calculate_accuracy(original, transcribed):
    """원본 문장과 변환된 텍스트의 정확도를 Levenshtein 거리로 계산"""
    if not transcribed:
        return 0.0
    distance = Levenshtein.distance(original, transcribed)
    max_length = max(len(original), len(transcribed))
    if max_length == 0:
        return 100.0 if original == transcribed else 0.0
    accuracy = (1 - distance / max_length) * 100
    return max(0.0, accuracy)

def play_audio(audio_file):
    """오디오 파일 재생 (sounddevice 사용)"""
    if not os.path.exists(audio_file):
        print(f"오류: 파일이 존재하지 않습니다: {audio_file}")
        return
    try:
        sample_rate, data = scipy.io.wavfile.read(audio_file)
        print(f"재생 중: {audio_file} (샘플레이트: {sample_rate}, 채널: {data.ndim})")
        sd.play(data, sample_rate)
        sd.wait()  # 재생 완료까지 대기
        print("재생 완료.")
    except Exception as e:
        print(f"재생 오류: {e}")

def main():
    # 문장과 진행 상황 로드
    sentences = load_sentences()
    total_sentences = len(sentences)
    current_index = load_progress()

    if current_index >= total_sentences:
        print("모든 문장이 녹음되었습니다!")
        return

    while current_index < total_sentences:
        sentence = sentences[current_index]
        print(f"\n문장 {current_index + 1}/{total_sentences}: {sentence}")

        # 고유한 녹음 파일 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join(OUTPUT_DIR, f"sentence_{current_index + 1}_{timestamp}.wav")

        # 녹음 수행
        audio_file = record_audio(audio_file)
        if not audio_file:
            print("녹음에 실패했습니다. 다시 시도하세요.")
            continue

        # 텍스트로 변환 및 정확도 계산
        transcribed_text = speech_to_text(audio_file)
        accuracy = calculate_accuracy(sentence, transcribed_text)
        print(f"\n변환된 텍스트: {transcribed_text}")
        print(f"정확도: {accuracy:.2f}%")

        # 옵션 제공
        while True:
            print("\n선택지:")
            print("1. 녹음 듣기")
            print("2. 재녹음")
            print("3. 다음 문장")
            print("4. 종료")
            choice = input("선택 (1-4): ").strip()

            if choice == '1':
                play_audio(audio_file)
            elif choice == '2':
                # 재녹음 전에 문장 다시 출력
                print(f"\n재녹음 문장 {current_index + 1}/{total_sentences}: {sentence}")
                # 현재 녹음 파일 삭제 후 재녹음
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                audio_file = record_audio(audio_file)
                if not audio_file:
                    print("녹음에 실패했습니다. 다시 시도하세요.")
                    continue
                transcribed_text = speech_to_text(audio_file)
                accuracy = calculate_accuracy(sentence, transcribed_text)
                print(f"\n변환된 텍스트: {transcribed_text}")
                print(f"정확도: {accuracy:.2f}%")
            elif choice == '3':
                # 다음 문장으로 이동
                save_progress(current_index + 1)
                current_index += 1
                break
            elif choice == '4':
                # 진행 상황 저장 후 종료
                save_progress(current_index)
                print("진행 상황을 저장하고 종료합니다...")
                return
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        if current_index >= total_sentences:
            print("모든 문장이 녹음되었습니다!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다. 진행 상황이 저장됩니다.")
    except Exception as e:
        print(f"오류 발생: {e}")