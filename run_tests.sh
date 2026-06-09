#!/bin/bash

# ==================================================
#  Скрипт для массового запуска тестов производительности
#  двойственного симплекс-метода
# ==================================================

# ---------- Настройки ----------
SOLVER_PATH="/home/vadim/Projects/dual_simplex_solver/build/solver"
TEST_DIR="/home/vadim/Projects/dual_simplex_solver/tests/test_data/kennington"
LOG_FILE="solver_runs_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT_SEC=1000
RUNS_PER_CONFIG=1

# MPI-параметры (только для solver_type=2)
MPI_PROCS=5            # число процессов MPI

# Параметры для параллельного метода (psi)
PSI_VALUES=(0.5 0.8 0.9 0.95 0.99)

# Список тестовых задач
PROBLEMS=(
    "ken-07.mps"
    "ken-11.mps"
    "osa-07.mps"
    "osa-14.mps"
    "osa-30.mps"
    "osa-60.mps"
    "pds-02.mps"
    # "pds-06.mps"
    "woodw.mps"
)

# ---------- Функции ----------
log_sep() {
    echo "============================================================" | tee -a "$LOG_FILE"
}

log_msg() {
    echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG_FILE"
}

# ---------- Очистка/создание лог-файла ----------
> "$LOG_FILE"
log_sep
log_msg "Запуск тестового прогона: $(date)"
log_msg "Исполняемый файл: $SOLVER_PATH"
log_msg "Каталог задач: $TEST_DIR"
log_msg "Таймаут (сек): $TIMEOUT_SEC"
log_msg "Повторов на конфигурацию: $RUNS_PER_CONFIG"
log_msg "MPI процессов (для типа 2): $MPI_PROCS"
log_sep

# ---------- Основной цикл ----------
for solver_type in 2; do
    if [ "$solver_type" -eq 0 ]; then
        SOLVER_NAME="Sequential"
    elif [ "$solver_type" -eq 1 ]; then
        SOLVER_NAME="CUDA"
    else
        SOLVER_NAME="MPI_Parallel"
    fi

    log_sep
    log_msg "===  Начало тестирования: $SOLVER_NAME  ==="
    log_sep

    for prob in "${PROBLEMS[@]}"; do
        prob_path="$TEST_DIR/$prob"

        # Пропускаем отсутствующие файлы
        if [ ! -f "$prob_path" ]; then
            log_msg "ПРОПУСК: файл $prob_path не найден"
            continue
        fi

        # Для MPI метода перебираем psi
        if [ "$solver_type" -eq 2 ]; then
            for psi in "${PSI_VALUES[@]}"; do
                log_sep
                log_msg "Задача: $prob | psi = $psi"
                log_sep

                for run in $(seq 1 $RUNS_PER_CONFIG); do
                    log_msg "--- Попытка $run из $RUNS_PER_CONFIG ---"
                    cmd="timeout $TIMEOUT_SEC mpirun -np $MPI_PROCS $SOLVER_PATH \"$prob_path\" $solver_type $psi"
                    log_msg "Команда: $cmd"
                    
                    # Запуск, stdout и stderr пишем в лог
                    eval "$cmd" >> "$LOG_FILE" 2>&1
                    exit_code=$?
                    if [ $exit_code -eq 124 ]; then
                        log_msg "РЕЗУЛЬТАТ: Превышен лимит времени (timeout)"
                    elif [ $exit_code -eq 0 ]; then
                        log_msg "РЕЗУЛЬТАТ: Завершено успешно"
                    else
                        log_msg "РЕЗУЛЬТАТ: Ошибка (код $exit_code)"
                    fi
                done
            done
        else
            # Для последовательного и CUDA метода psi не нужен
            log_sep
            log_msg "Задача: $prob"
            log_sep

            for run in $(seq 1 $RUNS_PER_CONFIG); do
                log_msg "--- Попытка $run из $RUNS_PER_CONFIG ---"
                cmd="timeout $TIMEOUT_SEC $SOLVER_PATH \"$prob_path\" $solver_type"
                log_msg "Команда: $cmd"
                
                eval "$cmd" >> "$LOG_FILE" 2>&1
                exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    log_msg "РЕЗУЛЬТАТ: Превышен лимит времени (timeout)"
                elif [ $exit_code -eq 0 ]; then
                    log_msg "РЕЗУЛЬТАТ: Завершено успешно"
                else
                    log_msg "РЕЗУЛЬТАТ: Ошибка (код $exit_code)"
                fi
            done
        fi
    done
done

log_sep
log_msg "Все тесты завершены: $(date)"
log_sep