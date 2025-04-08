import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import shutil

# Путь к весам модели SAM
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # Укажите путь к вашему файлу весов
MODEL_TYPE = "vit_h"  # Тип модели (vit_h, vit_l, vit_b)

# Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

# Загрузка модели SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)  # Используем доступное устройство

# Инициализация генератора масок
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=128)


# Функция для отображения масок на изображении
def show_anns(anns, image, output_dir, base_filename):
    if len(anns) == 0:
        print(f"Маски не найдены для изображения {base_filename}")
        return

    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        # Создаем цветную маску (например, красную)
        mask_image = np.zeros_like(image)
        mask_image[mask] = [255, 0, 0]  # Красный цвет для маски

        # Накладываем маску на исходное изображение
        overlaid_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

        # Сохраняем результат
        output_filename = os.path.join(output_dir, f"{base_filename}_mask_{i}.png")
        cv2.imwrite(output_filename, overlaid_image)
        print(f"Сохранено: {output_filename} (маска #{i})")


# Функция для сглаживания маски
def smooth_mask(mask, kernel_size=5):
    # Применяем Gaussian Blur для сглаживания границ
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    # Бинаризуем маску обратно (порог 0.5)
    smoothed_mask = (smoothed_mask > 0.5).astype(np.uint8)
    return smoothed_mask


# Функция для объединения выбранных масок и сохранения результата
def combine_masks(anns, selections, image, output_dir, final_mask_dir, base_filename, class_value):
    if len(anns) == 0:
        print(f"Маски не найдены для изображения {base_filename}")
        return

    # Создаем пустую маску для объединения
    combined_mask = np.zeros_like(anns[0]['segmentation'], dtype=bool)

    # Сначала обрабатываем положительные маски (без знаков)
    for idx, op in selections:
        if op == 'add' and 0 <= idx < len(anns):
            combined_mask = np.logical_or(combined_mask, anns[idx]['segmentation'])
        elif op == 'add':
            print(f"Маска с номером {idx} не существует, пропускаем.")

    # Затем обрабатываем инвертированные маски (с #)
    for idx, op in selections:
        if op == 'invert' and 0 <= idx < len(anns):
            # Добавляем всё кроме этой маски
            inverted_mask = ~anns[idx]['segmentation']
            combined_mask = np.logical_or(combined_mask, inverted_mask)
        elif op == 'invert':
            print(f"Маска с номером {idx} не существует, пропускаем.")

    # Наконец, вычитаем отрицательные маски (с -)
    for idx, op in selections:
        if op == 'subtract' and 0 <= idx < len(anns):
            # Вычитаем эту маску из объединенной
            combined_mask = np.logical_and(combined_mask, ~anns[idx]['segmentation'])
        elif op == 'subtract':
            print(f"Маска с номером {idx} не существует, пропускаем.")

    if not np.any(combined_mask):
        print("Объединённая маска пуста, ничего не сохраняем.")
        return

    # Создаем цветную маску (например, красную) для отображения
    mask_image = np.zeros_like(image)
    mask_image[combined_mask] = [255, 0, 0]  # Красный цвет для маски

    # Накладываем объединённую маску на исходное изображение
    overlaid_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    # Сохраняем результат объединённой маски с изображением
    output_filename = os.path.join(output_dir, f"{base_filename}_combined_mask.png")
    cv2.imwrite(output_filename, overlaid_image)
    print(f"Сохранено объединённое изображение: {output_filename}")

    # Создаем маску для сохранения в отдельный файл
    smoothed_mask = smooth_mask(combined_mask.astype(np.uint8))
    smoothed_mask_only = np.zeros_like(image, dtype=np.uint8)
    smoothed_mask_only[smoothed_mask > 0] = [class_value, class_value, class_value]

    # Сохраняем сглаженную маску в отдельный файл в папку final_masks
    mask_output_filename = os.path.join(final_mask_dir, f"{base_filename}_final_mask.png")
    cv2.imwrite(mask_output_filename, smoothed_mask_only)
    print(f"Сохранена финальная сглаженная маска: {mask_output_filename}")


# Функция для очистки папки
def clear_directory(directory):
    if os.path.exists(directory):
        # Удаляем папку и все её содержимое
        shutil.rmtree(directory)
    # Создаем папку заново
    os.makedirs(directory)
    print(f"Папка {directory} очищена и создана заново.")


# Функция для извлечения класса из имени файла
def extract_class_from_filename(filename):
    base_name = os.path.splitext(filename)[0]  # Убираем расширение
    try:
        class_value = int(base_name.split('_')[0])  # Извлекаем число до первого '_'
        return class_value
    except (ValueError, IndexError):
        print(f"Не удалось извлечь класс из имени файла {filename}. Используем значение по умолчанию 1.")
        return 1  # Значение по умолчанию, если класс не удалось извлечь

# Функция для проверки, существует ли финальная маска
def final_mask_exists(filename, final_mask_dir):
    base_filename = os.path.splitext(filename)[0]
    final_mask_path = os.path.join(final_mask_dir, f"{base_filename}_final_mask.png")
    return os.path.exists(final_mask_path)


def main():
    # Обрабатываем все изображения в текущей папке
    image_extensions = (".jpg", ".jpeg", ".png")  # Поддерживаемые форматы изображений
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(image_extensions):
            # Проверяем, существует ли финальная маска
            if final_mask_exists(filename, final_mask_dir):
                print(f"\nФинальная маска для изображения {filename} уже существует, пропускаем.")
                continue

            print(f"\nОбработка изображения: {filename}")

            # Очищаем папку output_masks перед обработкой нового изображения
            clear_directory(output_dir)

            # Извлекаем класс из имени файла
            class_value = extract_class_from_filename(filename)
            print(f"Извлечённый класс: {class_value}")

            # Загружаем изображение
            image_path = os.path.join(".", filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразуем BGR в RGB

            # Генерируем маски
            masks = mask_generator.generate(image)

            # Отображаем и сохраняем маски
            show_anns(masks, image, output_dir, os.path.splitext(filename)[0])

            if len(masks) == 0:
                continue

            # Запрашиваем у пользователя номера масок для объединения
            while True:
                try:
                    print(f"\nНайдено {len(masks)} масок (нумерация от 0 до {len(masks) - 1}).")
                    print("Используйте знак '-' перед номером маски для вычитания (например, -1)")
                    print("Используйте знак '#' перед номером маски для инвертирования (например, #2)")
                    user_input = input("Введите номера масок для объединения через пробел (или 'q' для пропуска): ")
                    if user_input.lower() == 'q':
                        break

                    # Парсим ввод пользователя
                    selections = []
                    for item in user_input.split():
                        if item.startswith('-'):  # Вычитание маски
                            try:
                                idx = int(item[1:])
                                selections.append((idx, 'subtract'))
                            except ValueError:
                                print(f"Ошибка: невозможно преобразовать '{item}' в номер маски.")
                                raise ValueError("Некорректный ввод")
                        elif item.startswith('#'):  # Инвертирование маски
                            try:
                                idx = int(item[1:])
                                selections.append((idx, 'invert'))
                            except ValueError:
                                print(f"Ошибка: невозможно преобразовать '{item}' в номер маски.")
                                raise ValueError("Некорректный ввод")
                        else:  # Обычное добавление маски
                            try:
                                idx = int(item)
                                selections.append((idx, 'add'))
                            except ValueError:
                                print(f"Ошибка: невозможно преобразовать '{item}' в номер маски.")
                                raise ValueError("Некорректный ввод")

                    # Проверяем, что все индексы масок корректны
                    all_indices_valid = all(0 <= idx < len(masks) for idx, _ in selections)
                    if all_indices_valid:
                        combine_masks(masks, selections, image, output_dir, final_mask_dir, os.path.splitext(filename)[0],
                                      class_value)
                        break
                    else:
                        print("Ошибка: один или несколько номеров масок вне допустимого диапазона. Попробуйте снова.")
                except ValueError as e:
                    print(f"Ошибка: {e}. Попробуйте снова.")

    print("Обработка завершена!")

if __name__ == "__main__":
    images_dir = r"./img/"
    # Создаем папки для сохранения результатов
    output_dir = "output_masks"  # Папка для промежуточных результатов
    final_mask_dir = "final_masks"  # Папка для финальных масок
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(final_mask_dir):
        os.makedirs(final_mask_dir)
    main()