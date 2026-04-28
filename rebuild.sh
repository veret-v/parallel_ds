# !/bin/bash
SOURCE_DIR="$(pwd)"
BUILD_DIR="${SOURCE_DIR}/build"

export MKLROOT=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl

# if [ -d "${BUILD_DIR}" ]; then
#     echo "Удаление существующей директории сборки..."
#     rm -rf "${BUILD_DIR}"
# fi

echo "Создание директории сборки..."
mkdir -p "$BUILD_DIR" ||  { echo "Ошибка создания каталога"; exit 1; }
cd "$BUILD_DIR" || { echo "Не удалось перейти в каталог сборки"; exit 1; }

echo "Конфигурация CMake..."
cmake .. || { echo "Ошибка конфигурации CMake"; exit 1; }

echo "Сборка проекта..."
make || { echo "Ошибка сборки"; exit 1; }

echo "Сборка успешно завершена."

# touch res.txt
# for file in ../tests/test_data/*
# do
#     timeout 1500s ./solver $file 1 >> res.txt
#     timeout 1500s ./solver $file 0 >> res.txt
# done

timeout 1500s ./solver ../tests/test_data/boeing1.mps 0
# ./solver ../tests/test_data/boeing1.mps


exit 0