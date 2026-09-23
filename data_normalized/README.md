# BDG2 (Kaggle-format) -> Du lieu chuan bi cho pipeline problem.md

Sinh boi `scripts/kaggle_ashrae/build_dataset.py` (mapping: `config/kaggle_ashrae_mapping.csv`).
CHI tao du lieu - khong huan luyen mo hinh du bao, khong giai bai toan toi uu (MILP/GA/PSO),
khong benchmark. Chay lai: `python scripts/kaggle_ashrae/build_dataset.py`.

## Nguon du lieu dau vao (khong tai lai tu Kaggle)

Da xac nhan voi nguoi dung dung file co san trong repo, cung schema voi bo tai truc tiep tu
Kaggle (row_id,building_id,meter,timestamp,meter_reading):

| Vai tro | File thuc te |
|---|---|
| train.csv | `data/building-data-genome-project-2/data/meters/kaggle/kaggle.csv` |
| building_metadata.csv | `data/building-data-genome-project-2/data/metadata/metadata.csv` (co them `building_id_kaggle`/`site_id_kaggle` de join) |
| weather_train.csv | `data/building-data-genome-project-2/data/weather/weather.csv` |

**Khac biet quan trong**: `kaggle.csv` trong repo chi phu **nam 2017** (theo README goc cua BDG2:
"the 2017 meter data that aligns with the Kaggle competition"), KHONG phai `train.csv` goc cua
cuoc thi (nam 2016). Cung schema va quy uoc meter code nen toan bo mapping/logic ap dung dung,
chi khac khoang thoi gian dữ lieu.

## Ket qua kiem tra Buoc 1

- Quy uoc meter code **0=electricity, 1=chilledwater, 2=steam, 3=hotwater** - xac nhan bang doi
  chieu so building_id/meter voi co Yes/NaN trong metadata.csv (thu tu electricity > chilledwater
  > steam > hotwater khop o ca hai nguon).
- **site_id_kaggle=0 (site "Panther") doc dien bang kBTU**: eui_proxy (kWh/ngay/sqft, chua quy
  doi) trung vi site 0 = 0.131, trung vi cac site khac = 0.027 -> ty le 4.83x, phu hop hien tuong
  kBTU (1 kWh = 3.412 kBTU). Da ap dung `kwh = meter_reading * 0.293071` cho **electricity (meter=0)
  tai site 0 mà thoi** - khong ap dung cho chilledwater (van de nay chi duoc xac nhan/yeu cau cho
  dien nang).
- Loc `building_metadata` con **828 toa nha** (Office 279 + Education 549) co du lieu Kaggle
  (`building_id_kaggle` khong rong).

## Output files

| File | Noi dung | Dinh dang |
|---|---|---|
| `telemetry_M1.parquet` | `power_active_kw`, `energy_active_kwh_total` + ngu canh | wide, 1 dong/(entity_id, ts) |
| `telemetry_M2.parquet` | `cooling_kw` + ngu canh | wide, 1 dong/(entity_id, ts) |
| `cooling_load_profile.parquet` | Chuoi `cooling_kw` that, 15 phut, cho toa nha bien do ngay-dem lon | dau vao THAT cho buoc toi uu |
| `equipment_params.csv` | 4 chiller gia dinh (Q, COP, duong cong EIR-FPLR) | tham so **DIEN HINH**, khong phai do dac |
| `tariff_params.csv` | Bieu gia EVN + khung gio TOU hien hanh (tra cuu web) | tham so tinh, co nguon |

### `telemetry_M1.parquet` / `telemetry_M2.parquet` - cot

`entity_id, ts, power_active_kw, energy_active_kwh_total, temperature, humidity, day_type,
tod_slot, site_id_kaggle, primaryspaceusage, unit_converted_kbtu_to_kwh` (M1) va tuong tu cho M2
(khong co `energy_active_kwh_total`, `unit_converted_kbtu_to_kwh`).

- **Luoi thoi gian = 1 gio** (nguyen ban Kaggle BDG2). problem.md muc 3.1 gia dinh luoi **15 phut**
  cho DMP that ("Resample ve luoi deu (15 phut voi DMP, 1 gio voi BDG2)") - hai file nay GIU
  NGUYEN 1 gio, KHONG noi suy. Chi `cooling_load_profile.parquet` co noi suy len 15 phut (xem duoi).
- `energy_active_kwh_total` la **counter tich luy MO PHONG** (`cumsum(power_active_kw)` tu 0 tai
  moi entity_id), phuc vu thu nghiem logic xu ly counter cua pipeline - KHONG PHAI counter thiet
  bi that (khong co reset/offset/gian doan).
- Dong bi thieu hoan toan (khong co ban ghi goc, ke ca meter_reading NaN trong nguon) duoc **loai
  khoi bang**, khong ghi dong NaN - xem % thieu trong bao cao cuoi script.
- `day_type`: 3 muc `ngay_lam_viec` / `cuoi_tuan` / `le`, suy tu ngay trong tuan + bang ngay nghi
  le Viet Nam **tu tao** (`src/normalization/calendar_vn_holidays.py`, khong phai lich nghi chinh
  thuc tung nam cua Chinh phu). KHAC voi 4 muc chuan DMP trong problem.md
  (workday/saturday/off/holiday) - o day gop Thu7+CN vao `cuoi_tuan`.
- `humidity` la **uoc luong** tu `dewTemperature` qua cong thuc Magnus-Tetens (Kaggle weather
  khong co cot RH truc tiep).

### `cooling_load_profile.parquet`

Chon toa nha co **median(max/min cooling_kw theo ngay) > 2.5** (toi thieu 30 ngay hop le, >=20
mau/ngay) tu `telemetry_M2` -> **218/349** toa nha co du lieu chilledwater dat nguong. Chuoi that
duoc noi suy tuyen tinh len buoc 15 phut **CHI trong tung khoang 1 gio thuc su lien tuc** (dung
`resample('15min').interpolate(limit=3)`) - KHONG lap cac khoang trong du lieu lon hon 1 gio (~75%
diem trong file la noi suy 15' giua 2 mau gio that, cot `is_interpolated` danh dau ro).

### `equipment_params.csv`

4 chiller **gia dinh dien hinh theo loai may** (Water-cooled Centrifugal/Screw, Air-cooled
Screw/Scroll), duong cong `P(x) = (Q_rated/COP_rated)*(a+b*x+c*x^2)` voi `a+b+c=1.0` (dam bao dung
diem dinh muc tai PLR=1). **KHONG PHAI so lieu do dac** - phai thay bang ho so thiet bi that khi
co du lieu DMP (problem.md muc 4.3).

### `tariff_params.csv`

- Bang gia theo cap dien ap/nhom khach hang: **Quyet dinh 1279/QD-BCT (09/05/2025, hieu luc
  10/05/2025)**, Bo Cong Thuong.
- Khung gio cao diem/binh thuong/thap diem: **Quyet dinh 963/QD-BCT (30/06/2026)** - khung gio
  **MOI**, thay doi so voi khung cu (9h30-11h30 & 17h-20h): cao diem **17h30-22h30 Thu2-Thu7**
  (Chu nhat KHONG co cao diem), thap diem **0h-6h moi ngay**.
- **Luu y quan trong**: nhom `hanh_chinh_su_nghiep_truong_hoc` (ap dung cho da so toa nha Education
  trong bo du lieu nay) la **gia don nhat theo cap dien ap, KHONG CO bieu gia TOU**. Voi cac toa
  nha Education, tin hieu chenh lech gia theo gio KHONG TON TAI o nhom gia nay - can xac nhan lai
  hop dong dien that truoc khi dung cho toi uu lich chay theo gia.
- Mac dinh de xuat cho toa nha Office (chua biet hop dong dien that): `customer_group=kinh_doanh`,
  `voltage_level=tu_22kV_tro_len`.

## Danh sach key trong problem.md ma Kaggle BDG2 KHONG CO (33 key)

Xem `config/kaggle_ashrae_mapping.csv` (trang_thai=`thieu_hoan_toan`) - chu yeu la: cong suat/nang
luong dien tung chiller (G3 trong problem.md), nhiet do/ap suat nuoc lanh cap-hoi, trang thai
thiet bi (chiller_state, compressor_status...), toan bo dien ap/dong dien tung pha M1, ZKTeco
(`record_count`, G4), nhat ky bao tri, va cac key trang thai dung chung (`reachable`, `fault`,
`mode`). Danh sach day du duoc in ra moi lan chay `build_dataset.py`.
