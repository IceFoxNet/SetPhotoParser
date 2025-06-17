from database import DBConnect
import os

while True:
    try:
        import math, requests, cv2, os, shutil, pathlib, gspread, yadisk
        import numpy as np
        from PIL import Image, ImageOps
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        package = e.msg.split()[-1][1:-1]
        os.system(f'python -m pip install {package}')
    else:
        break

workspace = pathlib.Path(__file__).parent.resolve()

# Функция для скачивания изображения
def download_image(image_url, save_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    }
    response = requests.get(image_url, headers=headers, proxies={'http': 'http://166.0.211.142:7576:user258866:pe9qf7'})
    if response.status_code == 200:
        with open(save_path, 'wb+') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Скачано: {save_path}")
        return True
    else:
        print(f"Не удалось скачать изображение: {image_url} (HTTP {response.status_code}) {response.text}")
        return False

# Функция для центрирования и обрезки изображения
def crop_and_center(image_path: str):
    try:
        img_color = Image.open(image_path)
        img_gray = img_color.convert("L")
        img_array = img_gray.load()
        width, height = img_gray.size
        threshold = 240

        # Поиск границ объекта
        x_min, y_min, x_max, y_max = width, height, 0, 0
        for y in range(height):
            for x in range(width):
                if img_array[x, y] < threshold:
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)

        if x_max == 0 or y_max == 0:
            print(f"Объект не найден на изображении {image_path}. Применяем жёсткие границы.")
            img_color = img_color.crop((50, 50, img_color.width - 50, img_color.height - 50))
        else:
            img_color = img_color.crop((x_min, y_min, x_max + 1, y_max + 1))

        # Добавление полей
        width, height = img_color.size
        max_dim = max(width, height)
        padding_top = (max_dim - height) // 2
        padding_bottom = max_dim - height - padding_top
        padding_left = (max_dim - width) // 2
        padding_right = max_dim - width - padding_left

        centered_img = ImageOps.expand(
            img_color,
            border=(padding_left, padding_top, padding_right, padding_bottom),
            fill=(255, 255, 255)
        )
        if 'BrickLink.png' in image_path: image_path = image_path.replace('BrickLink.png', 'BrickLink.jpg')
        centered_img.save(image_path)
        print(f"Объект выровнен и изображение сохранено: {image_path}")
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")

def is_box(img_path, vertex_threshold = 8, rectangularity_thresh = 0.8, solidity_thresh = 0.75):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 3)
    _, basic_thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    control_sum_adaptive = sum(np.sum(x) for x in adaptive_thresh)
    control_sum_basic = sum(np.sum(x) for x in basic_thresh)
    if (control_sum_adaptive / control_sum_basic) >= 0.22: thresh = basic_thresh
    else: thresh = adaptive_thresh
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    # Аппроксимация контура
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    vertices = len(approx)

    # Прямоугольность
    rect = cv2.minAreaRect(main_contour)
    min_rect_area = rect[1][0] * rect[1][1]
    contour_area = cv2.contourArea(main_contour)
    rectangularity = contour_area / min_rect_area if min_rect_area > 0 else 0

    # Солидность
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0

    # Компактность
    perimeter = cv2.arcLength(main_contour, True)
    compactness = (perimeter ** 2) / (4 * np.pi * contour_area) if contour_area > 0 else 0

    return (vertices <= vertex_threshold and rectangularity > rectangularity_thresh and solidity > solidity_thresh and compactness < 20)

# Функция для удаления дубликатов
def remove_duplicates(set_folder):
    images = sorted(os.listdir(set_folder))
    processed_images = set()

    for i, img1_name in enumerate(images):
        img1_path = os.path.join(set_folder, img1_name)
        if img1_name in processed_images:
            continue
        if is_box(img1_path):
            continue

        for img2_name in images[i + 1:]:
            img2_path = os.path.join(set_folder, img2_name)
            if img2_name in processed_images:
                continue
            if is_box(img2_path):
                continue

            try:

                ex1 = cv2.imread(img1_path)
                ex2 = cv2.imread(img2_path)

                if abs(ex1.shape[0]/ex2.shape[0] - ex1.shape[1]/ex2.shape[1]) < 0.01:
                    if ex1.shape[0] > ex2.shape[0]: ex2 = cv2.resize(ex2, ex1.shape[:-1])
                    else: ex1 = cv2.resize(ex1, ex2.shape[:-1])
                    gray1 = cv2.cvtColor(ex1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(ex2, cv2.COLOR_BGR2GRAY)
                    _, mask1 = cv2.threshold(gray1, 240, 255, cv2.THRESH_BINARY_INV)
                    _, mask2 = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY_INV)
                    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    obj_cont_1 = max(contours1, key=cv2.contourArea)
                    obj_cont_2 = max(contours2, key=cv2.contourArea)
                    template = np.zeros_like(mask1)
                    obj_mask_1 = cv2.drawContours(template.copy(), [obj_cont_1], -1, 255, thickness=cv2.FILLED)
                    obj_mask_2 = cv2.drawContours(template.copy(), [obj_cont_2], -1, 255, thickness=cv2.FILLED)
                    xor = cv2.bitwise_xor(obj_mask_1, obj_mask_2)
                    if (sum(np.sum(x)/255 for x in xor) / math.prod(xor.shape)) < 0.01:
                        size1 = os.path.getsize(img1_path)
                        size2 = os.path.getsize(img2_path)

                        if size1 >= size2:
                            os.remove(img2_path)
                            processed_images.add(img2_name)
                            print(f"Удалено {img2_name} (сохранено {img1_name})")
                        else:
                            os.remove(img1_path)
                            processed_images.add(img1_name)
                            print(f"Удалено {img1_name} (сохранено {img2_name})")
                            break

            except Exception as e:
                print(f"Ошибка при обработке пары {img1_name} и {img2_name}: {e}")

    print(f"Удаление дубликатов завершено. Обработанные файлы: {processed_images}")

# Функция для загрузки изображений
def scrape_images(set_number, yandex: yadisk.YaDisk, dbconn: DBConnect):
    base_folder = os.path.join(workspace, "images")
    raw_folder = os.path.join(base_folder, f'{set_number}-raw')

    os.makedirs(raw_folder, exist_ok=True)

    with sync_playwright() as p:
        driver = p.chromium.launch(proxy={
                'server': 'http://166.0.211.142:7576',
                'username': 'user258866',
                'password': 'pe9qf7'
        })
        page = driver.new_page()
        try:
            page.goto(f'https://bricker.ru/sets/{set_number}/')
            carousel = page.query_selector('#carousel')
            if carousel is not None:
                image_elements = carousel.query_selector_all('a')
                if not image_elements:
                    print(f"Полноразмерные изображения для набора {set_number} не найдены.")
                    return ''
                image_sources = ['https:' + img.get_attribute('href') for img in image_elements[1:]]
            else:
                page.goto(f'https://www.bricklink.com/v2/catalog/catalogitem.page?S={set_number}#T=P')
                carousel = page.query_selector('#_idtdThumbWrapper > div.pciThumbImgWindow')
                image_elements = carousel.query_selector_all('img')
                if not image_elements:
                    print(f"Полноразмерные изображения для набора {set_number} не найдены.")
                    return ''
                image_sources = ['https:' + img.get_attribute('src') for img in image_elements]

            for idx, img_src in enumerate(image_sources, start=1):
                full_image_url = img_src
                save_path = os.path.join(raw_folder, f"{idx}.png")
                if download_image(full_image_url, save_path):
                    crop_and_center(save_path)

            remove_duplicates(raw_folder)
            sort_edited_images(set_number, yandex, dbconn)
        except Exception as e:
            print(f"Ошибка: {e}")

def sort_edited_images(set_number, yandex: yadisk.YaDisk, dbconn: DBConnect):
    
    # Подготавливаем директории
    base_folder = os.path.join(workspace, "images")
    set_folder = os.path.join(base_folder, set_number)
    k_folder = os.path.join(base_folder, f'{set_number}-K')
    raw_folder = os.path.join(base_folder, f'{set_number}-raw')
    if os.path.exists(raw_folder):    
        os.makedirs(set_folder, exist_ok=True)
        os.makedirs(k_folder, exist_ok=True)
        
        # Сортируем все элементы
        all_items = sorted([os.path.join(raw_folder, image) for image in os.listdir(raw_folder)], key=is_box, reverse=True)
        box_counter = len(list(filter(is_box, all_items)))
        
        # Скачиваем изображение с BrickLink
        bricklink_path = os.path.join(raw_folder, 'BrickLink.png')
        bricklink_url = f"https://img.bricklink.com/ItemImage/SN/0/{set_number}-1.png"
        if download_image(bricklink_url, bricklink_path):
            crop_and_center(bricklink_path)
            os.remove(bricklink_path)
            bricklink_path = os.path.join(raw_folder, 'BrickLink.jpg')
        
        # Вносим корректировки
        if os.path.exists(bricklink_path):
            all_items = all_items[:box_counter] + [bricklink_path] + all_items[box_counter:]
        if box_counter == 2:
            all_items = all_items[:2][::-1] + all_items[2:]
        
        # Копируем элементы по папкам
        for idx in range(len(all_items)):
            shutil.copy(all_items[idx], os.path.join(k_folder, f'{idx+1}.jpg'))
        for idx in range(len(all_items)-box_counter):
            shutil.copy(all_items[idx+box_counter], os.path.join(set_folder, f'{idx+1}.jpg'))
        shutil.rmtree(raw_folder)
        
        try:
            yandex.remove(f'Авито/{set_number}')
        except:
            pass
        try:
            yandex.makedirs(f'Авито/{set_number}')
        except:
            pass
        for pic in os.listdir(set_folder):
            disk_path = f'Авито/{set_number}/{pic}'
            yandex.upload(os.path.join(set_folder, pic), disk_path, overwrite=True)
            yandex.publish(disk_path)
            media_url = yandex.get_meta(disk_path).public_url
            if media_url is not None:
                media_url = media_url.replace('yadi.sk', 'disk.yandex.ru')
            dbconn.delete_media(set_number, disk_path)
            dbconn.create_media(media_url, disk_path, set_number, f'ID-S-{set_number}-0-0', f'Карточка набора {pic}, без коробок, фотографии BrickLink + Bricker')
        shutil.rmtree(set_folder)
        
        try:
            yandex.remove(f'Авито/{set_number}-K')
        except:
            pass
        try:
            yandex.makedirs(f'Авито/{set_number}-K')
        except:
            pass
        for pic in os.listdir(k_folder):
            disk_path = f'Авито/{set_number}-K/{pic}'
            yandex.upload(os.path.join(k_folder, pic), disk_path, overwrite=True)
            yandex.publish(disk_path)
            media_url = yandex.get_meta(disk_path).public_url
            if media_url is not None:
                media_url = media_url.replace('yadi.sk', 'disk.yandex.ru')
            dbconn.delete_media(set_number, disk_path)
            dbconn.create_media(media_url, disk_path, str(set_number)+'-K', f'ID-S-{set_number}-0-0', f'Карточка набора {pic}, с коробками, фотографии BrickLink + Bricker')
        shutil.rmtree(k_folder)
        
        print(f'Фотографии по артикулу {set_number} расфасованы по директориям')
    else:
        print(f'Не удалось обнаружить директорию {raw_folder}, артикул пропущен')

# Подключение к Google Sheets и обработка артикулов
def main(start: int, end: int, setup: dict):
    # Подключаемся к таблице
    if start < 3: start = 3
    sheet: gspread.spreadsheet.Spreadsheet = setup.get('AutoloadSheet')
    yandex: yadisk.YaDisk = setup.get('YandexDisk')
    worksheet = sheet.worksheet("📦 Наборы")
    dbconn = DBConnect(setup.get('AppInfo'))

    # Получаем данные из диапазона D
    set_numbers = worksheet.range(f'D{start}:D{end}')


    # Обрабатываем каждый артикул
    for set_number in set_numbers:
        if dbconn.is_actual_media_generated(set_number.value) and dbconn.is_actual_media_generated(set_number.value + '-K'):
            print(f'Пропущен артикул {set_number.value}: актуальные карточки наборов уже сгенерированы')
            continue
        print(f"Обработка набора: {set_number.value}")
        scrape_images(set_number.value, yandex, dbconn)
    dbconn.close()

# Основной блок программы
if __name__ == "__main__":
    from Setup.setup import setup
    main(3, 8, setup)