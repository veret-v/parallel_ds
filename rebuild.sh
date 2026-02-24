#!/bin/bash
SOURCE_DIR="$(pwd)"
BUILD_DIR="${SOURCE_DIR}/build"

if [ -d "${BUILD_DIR}" ]; then
    echo "Удаление существующей директории сборки..."
    rm -rf "${BUILD_DIR}"
fi

echo "Создание директории сборки..."
mkdir -p "$BUILD_DIR" ||  { echo "Ошибка создания каталога"; exit 1; }
cd "$BUILD_DIR" || { echo "Не удалось перейти в каталог сборки"; exit 1; }

echo "Конфигурация CMake..."
cmake .. || { echo "Ошибка конфигурации CMake"; exit 1; }

echo "Сборка проекта..."
make || { echo "Ошибка сборки"; exit 1; }

echo "Сборка успешно завершена."
exit 0