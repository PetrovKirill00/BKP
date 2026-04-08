from constants import MOEX_SECURITIES_FILE, MOEX_CANDLES_DIR
import csv
from datetime import date, timedelta
import json
import requests
import time


def get_securities_json():
    """
    Получаю список ценных бумаг и сохраняю его в json
    """
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    r = requests.get(url, timeout=30)
    data = r.json()

    with open(MOEX_SECURITIES_FILE, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_secids():
    """
    Читаю полученный список ценных бумаг из json и выбираю все ценные бумаги по id.
    secid = security id, id ценной бумаги.
    """
    with open(MOEX_SECURITIES_FILE, 'r', encoding="utf-8") as f:
        data = json.load(f)

    columns = data["securities"]["columns"]
    rows = data["securities"]["data"]

    secid_idx = columns.index("SECID")
    sectype_idx = columns.index("SECTYPE")

    # Отфильтровываем обычные акции.
    # Привилегированные выкидываем.
    secids = [
        row[secid_idx]
        for row in rows
        if row[sectype_idx] == '1'
    ]

    return secids

def first_day_next_month(d):
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)

def iter_month_ranges(start_date, end_date):
    """
    Генерирует интервалы по месяцам в формате,
    который понимает MOEX candles endpoint
    """
    current = date(start_date.year, start_date.month, 1)

    while current <= end_date:
        next_month = first_day_next_month(current)
        chunk_start = max(current, start_date)
        chunk_end = min(next_month - timedelta(days=1), end_date)

        yield (
            f"{chunk_start.isoformat()} 00:00:00",
            f"{chunk_end.isoformat()} 23:59:59",
        )

        current = next_month


def fetch_candles_page(
        session,
        secid,
        dt_from,
        dt_till,
        start,
        retries = 5
):
    """
    Скачиваем одну страницу свечей для одного SECID
    """

    url = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"boards/TQBR/securities/{secid}/candles.json"
    )

    params = {
        "from": dt_from,
        "till": dt_till,
        "interval": 1,
        "start": start,
        "iss.meta": "off"
    }

    for attempt in range(retries):
        r = session.get(url, params=params, timeout=30)

        if r.status_code in (500, 502, 503, 504):
            if attempt == retries - 1:
                # Если это последняя попытка, то выбрасываем исключение
                r.raise_for_status()
            time.sleep(1.5 * (2 ** attempt))
            continue

        if r.status_code == 403:
            raise RuntimeError(
                f"{secid}: MOEX вернул 403. "
                f"Похоже, у этого ресурса есть ограничение доступа."
            )

        # raise_for_status() выдаёт исключение только если запрос был плохой
        # Если всё было норм, то он не выдаёт ничего
        r.raise_for_status()

        marker = r.headers.get("X-MicexPassport-Marker")
        payload = r.json()

        candles_block = payload.get("candles", {})
        columns = candles_block.get("columns", [])
        rows = candles_block.get("data", [])

        return columns, rows, marker

    raise RuntimeError(f"{secid}: не удалось скачать страницу свечей")

def download_candles_for_secid(secid, start_date, end_date):
    """
    Скачиваем данные по secid в данный промежуток.
    Данные грузятся по месяцам, учитывая пагинацию
    """
    out_dir = MOEX_CANDLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{secid}_TQBR_1m.csv"

    with requests.Session() as session:
        session.headers.update({"User-Agent": "moex-loader/1.0"})

        header_written = False

        with out_path.open('w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for dt_from, dt_till in iter_month_ranges(start_date, end_date):
                start = 0
                month_rows = 0

                while True:
                    columns, rows, marker = fetch_candles_page(
                        session,
                        secid,
                        dt_from,
                        dt_till,
                        start,
                    )

                    if marker == "denied" and start == 0 and not rows:
                        print(
                            f"{secid}: marker=denied, "
                            f"данные за {dt_from} .. {dt_till} не пришли"
                        )

                    if not rows:
                        break

                    if not header_written:
                        writer.writerow(columns)
                        header_written = True

                    writer.writerows(rows)
                    got = len(rows)
                    month_rows += got
                    start += got

                print(f"{secid}: {dt_from[:7]} -> {month_rows} rows")

    print(f"secid: сохранено в {out_path}")

def download_secids(secids):
    """
    Скачиваем котировки ценных бумаг для данного secid
    """
    start_date = date(2022, 1, 1)
    end_date = date.today()

    failed = set()

    for secid in secids:
        print("Загружаю котировки для акции", secid)
        try:
            download_candles_for_secid(
                secid,
                start_date,
                end_date
            )
        except Exception as e:
            failed.add(secid)
            print(f"{secid}: ошибка -> {e}")

    return list(failed)

def main():
    path = MOEX_SECURITIES_FILE
    if not path.exists():
        print("Скачиваю json с данными о ценных бумагах.")
        get_securities_json()
    else:
        print("Ценные бумаги уже загружены.")

    secids = read_secids()
    print("Ценные бумаги прочитаны из json.")

    failed_secids = download_secids(secids)
    download_secids(failed_secids)



if __name__ == "__main__":
    main()