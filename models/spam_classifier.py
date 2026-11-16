"""
Real-World NLP Spam Detection Module with Adaptive Split Inference & Quantization.
Supports:
  1. Trainable ModularTransformer (zero external HF weight requirement, works offline, save/load).
  2. Optional Pretrained HuggingFace DistilBERT Spam Classifier (when `transformers` is installed).
  3. Real dataset ingestion: Custom CSV/TSV, UCI SMS Spam Collection downloader, or built-in curated SMS benchmark.
  4. Full end-to-end evaluation combining NLP metrics (Accuracy, Precision, Recall, F1) with SimPy DES MEC latency & energy.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os
import re
import json
import time
import random
import urllib.request
import zipfile
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.transformer_model import ModularTransformer, TransformerConfig
from models.quantization import ActivationQuantizer
from models.split_inference import SplitInferenceExecutor
from simulation.environment import INDEX_TO_BITWIDTH, VALID_BITWIDTHS

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ==============================================================================
# 1. Realistic Curated SMS / Message Dataset (EN + RU) + UCI / CSV Loaders
# ==============================================================================

BUILTIN_SPAM_MESSAGES: List[str] = [
    "URGENT! You have won a 1000 cash prize. Claim now at http://bit.ly/claim-win before midnight!",
    "CONGRATULATIONS! Your number was selected for a FREE iPhone 15 Pro! Click http://promo-gift.xyz",
    "FreeMsg:Txt STOP to 85555 to cancel your weekly ringtone subscription 4.50/week.",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Exclusive loan approved! Get $5000 cash today with zero credit check: http://fast-cash-now.biz",
    "Your bank account has been temporarily suspended. Verify your card immediately at http://secure-verify.info",
    "HOT SINGLES in your area want to meet tonight! Reply YES or visit http://dating-link.cc",
    "FINAL NOTICE: Your parcel delivery failed due to unpaid fee $2.99. Pay now: http://track-parcel.me",
    "You have 1 unread voicemail from IRS Tax Office. Call 1-800-555-0199 immediately to avoid penalty.",
    "Get 80% OFF luxury watches today only! Limited stock! Order at http://cheap-watches.shop",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11 and send to 87575.",
    "Private! Your 2004 Account Statement for 07742676969 shows 786 unredeemed Bonus Points.",
    "Urgent! Call 09061743810 from landline. Your complimentary 4* Tenerife Holiday or 10000 cash awaits!",
    "Want explicit sex in 30 secs? Ring 02073162414 now! Costs 20p/min.",
    "Dear customer, you received a bonus transfer of $1,500 USDT. Withdraw at http://crypto-airdrop.io",
    "Win a brand new Tesla Model 3! Enter our free sweepstakes now at http://win-car-2026.com",
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour camera mobile for Free!",
    "Action Required: Unusual sign-in detected. Reset password now http://login-alert-security.net",
    "Cheap Viagra and pharmacy meds 70% discount! Fast shipping worldwide: http://rx-discount.org",
    "Double your Bitcoin in 24 hours guaranteed! Send 0.01 BTC to wallet address now!",
    "SMS SERVICES: for your inclusive text credits pls goto www.comuk.net login=3qxj9",
    "You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days.",
    "Claim your free $500 Amazon gift card! Only 3 spots left today: http://amz-rewards.vip",
    "IMPORTANT: Your Apple ID is locked. Unlock immediately at http://apple-support-verify.co",
    "Earn $300/day working from home! No experience needed. Message us on WhatsApp +155501234",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121.",
    "Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country Liverpool played in.",
    "Customer service announcement: You have a pending tax refund of $482.10. Claim at http://refund-gov.org",
    "Buy 1 get 3 FREE designer sunglasses! Offer expires in 2 hours: http://flash-sale-lux.com",
    "Reply WIN to 80085 for a chance to win 5000 cash every Friday! T&Cs apply 1.50/msg.",
    "ALERT! Someone accessed your PayPal from unknown device. Cancel transaction: http://pp-verify.cc",
    "Lose 10kg in 7 days with this secret pill! Doctors hate this trick: http://diet-miracle.biz",
    "Congratulations! You qualified for 0% APR credit card up to $15,000. Apply http://credit-fast.us",
    "We tried to contact you re your reply to our offer of 750 mins FREE any network mins.",
    "Your Netflix subscription payment declined. Update billing details: http://netflix-renew-pay.com",
    "FREE RINGTONE text FIRST to 87131 for a poly or text GET to 87131 for a true tone!",
    "You have been pre-approved for a mortgage relief grant of $12,000! Click http://grant-help.info",
    "Todays Vodafone numbers ending 4882 are selected to receive a $350 award. Call 09064019788",
    "Make $10,000 a month trading forex automatically with our AI bot! Sign up: http://fx-rich.io",
    "URGENT: Your package #US99482 is held at customs. Pay $3.50 clearance fee at http://customs-pay.net",
    "СРОЧНО! Вы выиграли 500 000 рублей в розыгрыше! Получите приз по ссылке: http://priz-bonus.ru",
    "Ваша банковская карта заблокирована! Срочно подтвердите данные на http://bank-secure-check.ru",
    "Одобрен кредит без справок и поручителей под 0%! Заберите деньги сейчас: http://zaym-online.biz",
    "Бесплатная подписка и 1000 бонусов только сегодня! Перейди по ссылке http://gift-free.cc",
    "ПОЗДРАВЛЯЕМ! Ваш номер стал победителем акции, позвоните 88005553535 для получения приза!",
    "Заработок от 15 000 руб в день на удалёнке без опыта! Пишите в телеграм @fast_money_bot",
    "Внимание! Попытка списания 12 400 руб. Отмените операцию по ссылке http://verify-card-ru.info",
    "Скидки до 90% на смартфоны только 2 часа! Успейте заказать на http://mega-sale-top.shop",
    "Вам начислен возврат налогов 28 450 руб. Оформите выплату на сайте http://nalog-vozvrat.org",
    "Пополни баланс на 100 рублей и получи 5000 рублей в подарок прямо сейчас! http://casino-win.bet",
    "URGENT! Final reminder to collect your £2000 supermarket voucher! Call 08712460324 now.",
    "Free 2nd week subscription to our mobile club! Text CLUB to 89990 to unlock unlimited downloads.",
    "Get rich quick! Guaranteed 400% return on crypto staking within 48 hours: http://token-moon.xyz",
    "Your mobile bill has an overpayment of £85.20. Enter bank details to refund: http://telco-refund.me",
    "BONUS ALERT: Deposit $10 and play with $200 FREE at LuckySpin Casino! http://luckyspin.win",
    "You've won a luxury weekend spa getaway for two! Dial 09058094565 to confirm booking.",
    "Secret investment strategy revealed by billionaires! Limited invites: http://wealth-insider.co",
    "WARNING: Your computer has 5 viruses! Download cleaner immediately: http://antivirus-fix-now.com",
    "Click here to claim your complimentary voucher for £150 at TopShop: http://voucher-free.net",
    "Last chance to get lifetime access to premium movies for just $1! http://stream-vip-deal.com",
    "Выплата компенсации от государства 45 000 руб доступна до конца дня: http://gos-viplata.info",
    "Супер-акция! Получи бесплатный билет и 3000 бонусов по промокоду WIN на http://promo-top.ru",
    "Срочное сообщение: истекает срок действия сим-карты, продлите по ссылке http://sim-update.cc",
    "Кредитная карта с лимитом 300 000 руб одобрена автоматически! Получить: http://card-express.biz",
    "Уважаемый абонент, на ваш счёт зачислен выигрыш 100 000 руб, оплатите комиссию на http://pay-win.ru",
    "Hi babe how r u? Wanna chat with hot girls tonight? Call 09099726481 only 1.50/min!",
    "Congratulations! U won 2 tickets to the VIP concert! Text YES to 82277 now to claim!",
    "Special offer: Order Rolex replica watches at 90% off! Visit http://lux-timepieces.shop",
    "Your UPS shipment could not be delivered. Reschedule and pay $1.99 fee: http://ups-redeliver.info",
    "You have received a secure voice message from Bank Fraud Dept. Click http://fraud-alert-check.com",
    "Txt MUSIC to 85023 to get 10 FREE MP3 ringtones every week! Only £3/wk.",
    "FREE payday loan! Instant approval up to $3000 directly to your bank: http://cash24hr.biz",
    "Notice: Your email storage is 99% full. Upgrade for free or verify account: http://mail-quota.net",
    "Win £1000 shopping spree! Just reply SHOP to 84122 right now! Standard rates apply.",
    "Cheap flights to Dubai $49! Book within the next 30 minutes at http://fly-cheap-promo.com",
    "Срочно погасите задолженность или получите рефинансирование под 1% на http://debt-zero.ru",
    "Ваш аккаунт Госуслуг взломан? Срочно смените пароль через сайт http://login-check-sec.ru",
    "Дарим 10 000 рублей новым игрокам! Регистрация за 1 минуту: http://bet-bonus777.com",
    "Распродажа конфиската электроники за 10% цены! Только сегодня http://stock-confiscat.shop",
    "Вы попали в список получателей приза! Отправьте код 774 на короткий номер 4455 бесплатно."
]

BUILTIN_HAM_MESSAGES: List[str] = [
    "Hey, are we still meeting for coffee at 3pm near the library?",
    "I'll be home in about 20 minutes, do you want me to pick up some milk and bread?",
    "Thanks for sending over the lecture notes, that really helped me prepare for tomorrow's exam.",
    "Sorry I missed your call, I was in a team meeting. Can I call you back after 5?",
    "Happy birthday! Wishing you health, happiness, and an amazing year ahead!",
    "Don't forget we have a dentist appointment on Thursday morning at 9:30 AM.",
    "Did you finish implementing the PyTorch dataloader for the NLP assignment?",
    "Let's grab lunch at the campus cafeteria once the physics lab ends.",
    "The train is delayed by 10 minutes due to rain, so I might arrive slightly late.",
    "Can you review my pull request when you have a spare moment this afternoon?",
    "Mom asked if we are visiting this weekend for Sunday dinner.",
    "Just checked the weather forecast, it looks sunny all week so bring sunglasses.",
    "Please send me the updated PDF slides before the presentation starts at noon.",
    "I left my jacket in your car yesterday, can I grab it from you tomorrow?",
    "Sounds good! See you at the main entrance at 6:15.",
    "Wait, what chapter did the professor say will be covered on the midterm?",
    "Sure, I can help you debug the Python environment setup tonight on Zoom.",
    "We booked the conference room on the 4th floor for our sprint planning session.",
    "Have a safe flight! Text me as soon as you land in Berlin.",
    "I'm running a bit behind schedule, please start the call without me.",
    "Could you pick up some fresh tomatoes and pasta on your way back from work?",
    "Great job on the seminar presentation today, the charts looked super clear.",
    "Let me know when you get home safely so I don't worry.",
    "I watched that documentary you recommended last night, it was fascinating.",
    "Remember to bring your student ID to get into the computer lab.",
    "No worries at all, we can reschedule the gym session to Friday evening.",
    "The parcel with the new router arrived at the front desk, I already picked it up.",
    "What time does the grocery store close on Sundays?",
    "Yes, I read the paper on split inference and quantization, let's discuss the math tomorrow.",
    "Coffee machine in the office is broken again, bring tea from home.",
    "Thanks for the birthday wishes! It was great catching up with everyone.",
    "Can you share the Wi-Fi password for the guest network?",
    "I'll upload the dataset and evaluation metrics script to the shared drive tonight.",
    "My battery is at 4 percent, texting you quickly before the phone dies.",
    "Sure, let's split the Uber fare when we leave the concert.",
    "Did you pay the electricity bill for this month or should I transfer my half?",
    "The cat knocked over a plant again while we were out.",
    "I'm at the supermarket right now, need anything else besides apples and yogurt?",
    "Let's take a quick walk in the park after finishing the report.",
    "Congrats on passing the final defense! We should celebrate this Friday.",
    "Привет! Мы сегодня встречаемся у метро в 18:30 как договаривались?",
    "Я буду дома минут через двадцать, нужно купить хлеб или молоко по пути?",
    "Спасибо за конспекты по машинному обучению, очень выручили перед зачётом!",
    "Извини, не мог ответить, был на паре. Наберу тебе после пяти вечера.",
    "С днём рождения! Желаю успехов в учёбе, крепкого здоровья и отличного настроения!",
    "Не забудь, пожалуйста, завтра утром в 10:00 у нас созвон с научным руководителем.",
    "Ты уже успел запустить тесты симулятора на локальном компьютере?",
    "Пойдём пообедаем в столовой после окончания лабораторной работы?",
    "Электричка задерживается на десять минут, поэтому немного опоздаю на встречу.",
    "Посмотри, пожалуйста, новую версию презентации к диплому, я скинул в чат.",
    "Мама спрашивала, приедем ли мы к ним на дачу в эти выходные.",
    "По прогнозу завтра обещают дождь, лучше возьми с собой зонтик.",
    "Отправь мне, пожалуйста, актуальный файл с результатами эксперимента в формате CSV.",
    "Я случайно оставил тетрадь у тебя на столе, отдашь завтра перед лекцией?",
    "Договорились! Встретимся у главного входа в университет ровно в два.",
    "Подскажи, какие именно главы учебника будут на завтрашнем коллоквиуме?",
    "Давай вечером созвонимся, помогу настроить виртуальное окружение Python.",
    "Мы забронировали аудиторию 305 для репетиции защиты проекта.",
    "Удачного полёта! Напиши сообщение, как только самолёт приземлится.",
    "Я немного задерживаюсь в пробке, начинайте обсуждение без меня.",
    "Купи, пожалуйста, по дороге домой сыр и помидоры на салат.",
    "Отличный доклад сегодня получился, графики квантования смотрелись очень убедительно.",
    "Напиши, как доберёшься до дома, чтобы я не переживал.",
    "Посмотрел фильм, который ты советовал на прошлой неделе — реально крутой.",
    "Не забудь пропуск в корпус, сегодня на охране строго проверяют документы.",
    "Ничего страшного, можем перенести тренировку в зале на пятницу вечером.",
    "Заказ из пункта выдачи я уже забрал, всё пришло целым.",
    "Подскажите, до скольки сегодня работает библиотека на втором этаже?",
    "Да, я прочитал статью про адаптивное разделение нейросетей, обсудим завтра детали.",
    "Кулер в коридоре опять пустой, возьми бутылку воды с собой.",
    "Большое спасибо за поздравление! Очень приятно было увидеться.",
    "Можешь скинуть пароль от лабораторного роутера, ноутбук не подключается.",
    "Я загрузил графики точности и задержки в общую папку проекта.",
    "Телефон почти разрядился, осталось три процента, доеду — напишу.",
    "Конечно, давай разделим стоимость такси пополам после концерта.",
    "Ты уже оплатил интернет за этот месяц или мне перевести свою часть?",
    "Кот снова скинул кружку со стола, пока нас не было дома.",
    "Я сейчас в магазине, нужно что-то ещё кроме яблок, молока и овсянки?",
    "Давай немного прогуляемся по парку после того, как допишем отчёт.",
    "Поздравляю с успешной предзащитой! Обязательно отметим на выходных."
]


class SpamDatasetLoader:
    """Loads curated built-in SMS spam/ham dataset, custom CSV/TSV files, or UCI SMS Spam Collection."""

    UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

    @staticmethod
    def load_builtin_dataset(
        test_ratio: float = 0.25,
        seed: int = 42
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Returns stratified (train_texts, train_labels, test_texts, test_labels).
        Labels: 0 = HAM (Not Spam), 1 = SPAM.
        """
        rng = random.Random(seed)
        spams = [(txt, 1) for txt in BUILTIN_SPAM_MESSAGES]
        hams = [(txt, 0) for txt in BUILTIN_HAM_MESSAGES]

        rng.shuffle(spams)
        rng.shuffle(hams)

        n_test_spam = max(1, int(len(spams) * test_ratio))
        n_test_ham = max(1, int(len(hams) * test_ratio))

        test_items = spams[:n_test_spam] + hams[:n_test_ham]
        train_items = spams[n_test_spam:] + hams[n_test_ham:]

        rng.shuffle(train_items)
        rng.shuffle(test_items)

        train_texts = [x[0] for x in train_items]
        train_labels = [x[1] for x in train_items]
        test_texts = [x[0] for x in test_items]
        test_labels = [x[1] for x in test_items]
        return train_texts, train_labels, test_texts, test_labels

    @staticmethod
    def parse_csv_or_tsv_bytes(
        raw_bytes: bytes,
        test_ratio: float = 0.20,
        seed: int = 42
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Parses uploaded CSV or UCI tab-separated SMSSpamCollection file.
        Supports columns: (label, text), (v1, v2), or 2-column TSV without header.
        """
        import csv
        text_stream = raw_bytes.decode("utf-8", errors="replace").splitlines()
        items: List[Tuple[str, int]] = []

        # Detect TSV without header (UCI SMSSpamCollection format: "ham\tmessage...")
        if text_stream and "\t" in text_stream[0] and not text_stream[0].lower().startswith("label\t"):
            for line in text_stream:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    lbl_str = parts[0].strip().lower()
                    msg = parts[1].strip()
                    if msg:
                        lbl = 1 if lbl_str in ("1", "spam", "true", "yes") else 0
                        items.append((msg, lbl))
        else:
            reader = csv.DictReader(text_stream)
            fieldnames = [f.lower().strip() for f in (reader.fieldnames or [])]
            for row in reader:
                row_lower = {k.lower().strip(): v for k, v in row.items() if k is not None}
                lbl_val = row_lower.get("label") or row_lower.get("v1") or row_lower.get("class") or row_lower.get("target")
                txt_val = row_lower.get("text") or row_lower.get("v2") or row_lower.get("message") or row_lower.get("sms")
                if lbl_val is not None and txt_val:
                    lbl = 1 if str(lbl_val).strip().lower() in ("1", "spam", "true", "yes") else 0
                    items.append((str(txt_val).strip(), lbl))

        if len(items) < 10:
            raise ValueError("Dataset must contain at least 10 valid labeled messages.")

        rng = random.Random(seed)
        spams = [x for x in items if x[1] == 1]
        hams = [x for x in items if x[1] == 0]
        rng.shuffle(spams)
        rng.shuffle(hams)

        n_test_s = max(1, int(len(spams) * test_ratio)) if spams else 0
        n_test_h = max(1, int(len(hams) * test_ratio)) if hams else 0

        test_items = spams[:n_test_s] + hams[:n_test_h]
        train_items = spams[n_test_s:] + hams[n_test_h:]
        rng.shuffle(train_items)
        rng.shuffle(test_items)

        return (
            [x[0] for x in train_items],
            [x[1] for x in train_items],
            [x[0] for x in test_items],
            [x[1] for x in test_items],
        )

    @classmethod
    def download_uci_sms_spam(
        cls,
        save_dir: str = "data",
        test_ratio: float = 0.20,
        seed: int = 42
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Downloads UCI SMS Spam Collection (5,574 real English SMS) if online."""
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, "SMSSpamCollection")

        if not os.path.exists(local_path):
            req = urllib.request.Request(cls.UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                with zf.open("SMSSpamCollection") as src, open(local_path, "wb") as dst:
                    dst.write(src.read())

        with open(local_path, "rb") as f:
            raw_bytes = f.read()
        return cls.parse_csv_or_tsv_bytes(raw_bytes, test_ratio=test_ratio, seed=seed)


# ==============================================================================
# 2. Subword & Lexical Feature Tokenizer for ModularTransformer
# ==============================================================================

class SpamTextTokenizer:
    """
    Lightweight, saveable tokenizer for Transformer text classification.
    Maps URLs, phone numbers, currency symbols, exclamation marks, words, and character 3-grams
    to handle out-of-vocabulary words reliably in both English and Russian.
    Special IDs: 0 = [PAD], 1 = [UNK], 2 = [CLS]
    """

    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2

    def __init__(self, max_vocab_size: int = 8000, max_seq_len: int = 96):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.token_to_id: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}

    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        txt = text.lower().strip()
        # Normalize salient spam indicators before splitting
        txt = re.sub(r"https?://\S+|www\.\S+", " <url> ", txt)
        txt = re.sub(r"[£$€₽]", " <currency> ", txt)
        txt = re.sub(r"\b\d{4,}\b", " <longnum> ", txt)
        txt = re.sub(r"\b\d+\b", " <num> ", txt)
        txt = re.sub(r"!+", " ! ", txt)
        raw_words = re.findall(r"<url>|<currency>|<longnum>|<num>|!|[a-zа-яё0-9]+", txt)

        # Include stems (prefix 4 chars) to handle Russian/English morphology cleanly
        expanded: List[str] = []
        for w in raw_words:
            expanded.append(w)
            if not w.startswith("<") and len(w) >= 5:
                expanded.append(f"__pre_{w[:4]}")
        return expanded

    def fit(self, texts: List[str]) -> None:
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in self._extract_tokens(t):
                freq[tok] = freq.get(tok, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
        for tok, _ in sorted_tokens:
            if len(self.token_to_id) >= self.max_vocab_size:
                break
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)

    def encode(
        self,
        text: str,
        pad_to_max: bool = False,
        max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes single string into (input_ids [1, L], additive_attention_mask [1, 1, 1, L]).
        When pad_to_max=False, keeps the actual message token length for accurate MEC payload size!
        """
        cap = max_len or self.max_seq_len
        tokens = self._extract_tokens(text)[: cap - 1]
        ids = [self.CLS_ID] + [self.token_to_id.get(tok, self.UNK_ID) for tok in tokens]

        if pad_to_max and len(ids) < cap:
            ids = ids + [self.PAD_ID] * (cap - len(ids))

        input_ids = torch.tensor([ids], dtype=torch.long)
        # Additive attention mask compatible with MultiHeadSelfAttention: 0.0 keep, -1e9 mask
        mask = torch.where(input_ids == self.PAD_ID, -1e9, 0.0).unsqueeze(1).unsqueeze(1)
        return input_ids, mask

    def encode_batch(self, texts: List[str], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cap = max_len or self.max_seq_len
        encoded = [self.encode(t, pad_to_max=False, max_len=cap)[0].squeeze(0).tolist() for t in texts]
        batch_max = min(cap, max(len(x) for x in encoded))
        padded = [x[:batch_max] + [self.PAD_ID] * max(0, batch_max - len(x)) for x in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        mask = torch.where(input_ids == self.PAD_ID, -1e9, 0.0).unsqueeze(1).unsqueeze(1)
        return input_ids, mask

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"max_vocab_size": self.max_vocab_size, "max_seq_len": self.max_seq_len, "vocab": self.token_to_id},
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path: str) -> "SpamTextTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data["max_vocab_size"], max_seq_len=data["max_seq_len"])
        tok.token_to_id = {k: int(v) for k, v in data["vocab"].items()}
        return tok


class _TokenizedDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


# ==============================================================================
# 3. Classification Metrics Helper (no sklearn required)
# ==============================================================================

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, float]:
    """Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrix counts."""
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_samples": total
    }


# ==============================================================================
# 4. Trainable & Split-Capable Modular Spam Classifier
# ==============================================================================

@dataclass
class SpamInferenceResult:
    """Single message classification result combining NLP and MEC Split metrics."""
    text: str
    predicted_label: str       # "SPAM" or "HAM"
    predicted_class: int       # 1 = SPAM, 0 = HAM
    spam_probability: float
    ham_probability: float
    num_tokens: int
    split_point: int
    bitwidth: int
    transmitted_bytes: int
    compression_ratio: float
    cosine_similarity: float
    top1_match_vs_fp32: bool
    t_device_measured_ms: float
    t_mec_measured_ms: float


class ModularSpamClassifier:
    """
    Production-ready Transformer Spam Classifier built on ModularTransformer + SplitInferenceExecutor.
    Trains end-to-end on real labeled SMS/text datasets, saves/loads weights, and executes real
    split inference with FP32/INT16/INT8/INT4 activation quantization.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 96,
        checkpoint_dir: str = "saved_models/spam_transformer"
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.checkpoint_dir = checkpoint_dir

        self.tokenizer = SpamTextTokenizer(max_vocab_size=8000, max_seq_len=max_seq_len)
        self.config = TransformerConfig(
            model_name=f"ModularSpamTransformer-L{num_layers}-d{hidden_dim}",
            vocab_size=8000,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=hidden_dim * 2,
            max_position_embeddings=max_seq_len + 16,
            dropout_prob=0.1,
            num_classes=2
        )
        self.model = ModularTransformer(self.config)
        self.executor = SplitInferenceExecutor(self.model)
        self.is_trained: bool = False
        self.training_history: List[Dict[str, float]] = []

    def train_on_dataset(
        self,
        train_texts: List[str],
        train_labels: List[int],
        epochs: int = 14,
        batch_size: int = 16,
        lr: float = 8e-4,
        seed: int = 42,
        progress_callback: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """Fine-tunes ModularTransformer end-to-end on labeled spam/ham messages."""
        torch.manual_seed(seed)
        self.tokenizer.fit(train_texts)

        # Rebuild model matched cleanly to actual vocab size for compact footprint
        vocab_size = max(128, len(self.tokenizer.token_to_id))
        self.config.vocab_size = vocab_size
        self.model = ModularTransformer(self.config)
        self.executor = SplitInferenceExecutor(self.model)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        ds = _TokenizedDataset(train_texts, train_labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.training_history = []
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_texts, batch_labels in loader:
                input_ids, attn_mask = self.tokenizer.encode_batch(list(batch_texts))
                labels_t = torch.as_tensor(batch_labels, dtype=torch.long)

                optimizer.zero_grad()
                hidden = self.model.embedding(input_ids)
                for layer in self.model.layers:
                    hidden = layer(hidden, attention_mask=attn_mask)
                cls_vec = hidden[:, 0, :]
                logits = self.model.classifier(cls_vec)

                loss = criterion(logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item()) * len(labels_t)
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == labels_t).sum().item())
                total += len(labels_t)

            ep_loss = total_loss / max(1, total)
            ep_acc = correct / max(1, total)
            self.training_history.append({"epoch": epoch, "loss": round(ep_loss, 4), "train_acc": round(ep_acc, 4)})
            if progress_callback:
                progress_callback(epoch, epochs, ep_loss, ep_acc)

        self.model.eval()
        self.executor = SplitInferenceExecutor(self.model)
        self.is_trained = True
        return self.training_history

    def save(self) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(self.checkpoint_dir, "tokenizer.json"))
        meta = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.config.vocab_size,
            "training_history": self.training_history
        }
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "model.pt"))
        return self.checkpoint_dir

    def load(self) -> bool:
        cfg_path = os.path.join(self.checkpoint_dir, "config.json")
        tok_path = os.path.join(self.checkpoint_dir, "tokenizer.json")
        wt_path = os.path.join(self.checkpoint_dir, "model.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(tok_path) and os.path.exists(wt_path)):
            return False

        with open(cfg_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_layers = meta["num_layers"]
        self.hidden_dim = meta["hidden_dim"]
        self.num_heads = meta["num_heads"]
        self.max_seq_len = meta["max_seq_len"]
        self.training_history = meta.get("training_history", [])
        self.tokenizer = SpamTextTokenizer.load(tok_path)

        self.config = TransformerConfig(
            model_name=f"ModularSpamTransformer-L{self.num_layers}-d{self.hidden_dim}",
            vocab_size=meta["vocab_size"],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_dim=self.hidden_dim * 2,
            max_position_embeddings=self.max_seq_len + 16,
            dropout_prob=0.1,
            num_classes=2
        )
        self.model = ModularTransformer(self.config)
        self.model.load_state_dict(torch.load(wt_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.executor = SplitInferenceExecutor(self.model)
        self.is_trained = True
        return True

    @torch.no_grad()
    def predict_split(
        self,
        text: str,
        split_point: int = 2,
        bitwidth: int = 8
    ) -> SpamInferenceResult:
        """Executes real split inference + quantization on an arbitrary text message."""
        self.model.eval()
        input_ids, attn_mask = self.tokenizer.encode(text, pad_to_max=False)
        num_tokens = int(input_ids.shape[1])

        out = self.executor.execute(
            input_ids=input_ids,
            split_point=split_point,
            bitwidth=bitwidth,
            quantization_enabled=(bitwidth < 32),
            attention_mask=attn_mask,
            compute_ground_truth_comparison=True
        )

        probs = F.softmax(out.logits, dim=-1)[0]
        ham_prob = float(probs[0].item())
        spam_prob = float(probs[1].item())
        pred_cls = int(out.predicted_class)
        pred_lbl = "SPAM" if pred_cls == 1 else "HAM"

        return SpamInferenceResult(
            text=text,
            predicted_label=pred_lbl,
            predicted_class=pred_cls,
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            num_tokens=num_tokens,
            split_point=out.split_point,
            bitwidth=out.bitwidth,
            transmitted_bytes=out.transmitted_bytes,
            compression_ratio=out.compression_ratio,
            cosine_similarity=out.quality_metrics.get("cosine_similarity", 1.0),
            top1_match_vs_fp32=bool(out.quality_metrics.get("top1_match", 1.0) >= 0.5),
            t_device_measured_ms=round(out.t_device_measured_s * 1000.0, 3),
            t_mec_measured_ms=round(out.t_mec_measured_s * 1000.0, 3)
        )


# ==============================================================================
# 5. Optional Pretrained HuggingFace DistilBERT Split Classifier
# ==============================================================================

class HuggingFaceSplitSpamClassifier:
    """
    Executes real split inference on a pretrained HuggingFace DistilBERT spam model
    (e.g. 'mrm8488/distilbert-finetuned-sms-spam-detection'), inserting ActivationQuantizer
    physically between transformer blocks k-1 and k.
    """

    DEFAULT_HF_MODEL = "mrm8488/distilbert-finetuned-sms-spam-detection"

    def __init__(self, model_name: str = DEFAULT_HF_MODEL):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("`transformers` package is not installed. Run: pip install transformers")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.quantizer = ActivationQuantizer()
        self.num_layers = len(self.model.distilbert.transformer.layer)
        self.is_trained = True

    @torch.no_grad()
    def predict_split(
        self,
        text: str,
        split_point: int = 2,
        bitwidth: int = 8
    ) -> SpamInferenceResult:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        num_tokens = int(input_ids.shape[1])

        k = max(0, min(self.num_layers, split_point))

        # FP32 baseline logits
        fp32_logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
        fp32_cls = int(torch.argmax(fp32_logits, dim=-1)[0].item())

        t_device = 0.0
        t_mec = 0.0
        tx_bytes = 0
        comp_ratio = 1.0

        # Prepare DistilBERT attention mask format
        bool_mask = attn_mask.bool()

        if k == self.num_layers:
            t0 = time.perf_counter()
            logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
            t_device = time.perf_counter() - t0
            used_bits = 32
        elif k == 0:
            tx_bytes = int(input_ids.numel() * 4)
            t0 = time.perf_counter()
            logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
            t_mec = time.perf_counter() - t0
            used_bits = 32
        else:
            t0 = time.perf_counter()
            hidden = self.model.distilbert.embeddings(input_ids)
            for i in range(k):
                hidden = self.model.distilbert.transformer.layer[i](hidden, attn_mask=bool_mask)[0]
            t_device = time.perf_counter() - t0

            q_res = self.quantizer.quantize(hidden, bitwidth=bitwidth, enabled=(bitwidth < 32))
            tx_bytes = q_res.compressed_size_bytes
            comp_ratio = q_res.compression_ratio
            deq_hidden = self.quantizer.dequantize(q_res, as_torch_tensor=True).to(hidden.device)

            t1 = time.perf_counter()
            cur = deq_hidden
            for i in range(k, self.num_layers):
                cur = self.model.distilbert.transformer.layer[i](cur, attn_mask=bool_mask)[0]
            pooled = cur[:, 0]
            pooled = self.model.pre_classifier(pooled)
            pooled = nn.ReLU()(pooled)
            logits = self.model.classifier(pooled)
            t_mec = time.perf_counter() - t1
            used_bits = bitwidth

        q_metrics = self.quantizer.evaluate_quality(fp32_logits, logits)
        probs = F.softmax(logits, dim=-1)[0]
        ham_prob = float(probs[0].item())
        spam_prob = float(probs[1].item())
        pred_cls = int(torch.argmax(logits, dim=-1)[0].item())

        return SpamInferenceResult(
            text=text,
            predicted_label="SPAM" if pred_cls == 1 else "HAM",
            predicted_class=pred_cls,
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            num_tokens=num_tokens,
            split_point=k,
            bitwidth=used_bits,
            transmitted_bytes=tx_bytes,
            compression_ratio=round(comp_ratio, 2),
            cosine_similarity=round(q_metrics["cosine_similarity"], 4),
            top1_match_vs_fp32=(pred_cls == fp32_cls),
            t_device_measured_ms=round(t_device * 1000.0, 3),
            t_mec_measured_ms=round(t_mec * 1000.0, 3)
        )


# ==============================================================================
# 6. End-to-End Evaluation on Real Dataset with MECSimulationEnv (SimPy DES)
# ==============================================================================

def evaluate_spam_dataset_with_policy(
    classifier: Any,
    policy: Any,
    env: Any,
    test_texts: List[str],
    test_labels: List[int],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs every real test message through BOTH:
      1) Policy action selection (k, b, f) on MECSimulationEnv (SimPy DES),
      2) Real PyTorch Split Inference + Activation Quantization on the message text!
    Returns exact NLP metrics (Accuracy, Precision, Recall, F1) + MEC SimPy latency/energy metrics.
    """
    obs, info = env.reset(seed=seed)
    policy.reset()

    y_true: List[int] = []
    y_pred: List[int] = []
    agreements: int = 0
    latencies_ms: List[float] = []
    energies_mj: List[float] = []
    tx_bytes_list: List[int] = []
    split_points: List[int] = []
    bitwidths: List[int] = []
    sla_hits: int = 0
    sample_logs: List[Dict[str, Any]] = []

    for idx, (text, label) in enumerate(zip(test_texts, test_labels)):
        # Override current task sequence length with the REAL tokenized length of this SMS
        tok_ids, _ = classifier.tokenizer.encode(text) if hasattr(classifier, "tokenizer") and hasattr(classifier.tokenizer, "encode") else ([], None)
        real_seq_len = max(8, int(tok_ids.shape[1]) if hasattr(tok_ids, "shape") else len(text.split()))
        if env.current_task is not None:
            env.current_task.sequence_length = real_seq_len
            obs = env._get_obs()
            info["seq_len"] = real_seq_len

        action = policy.select_action(obs, info)
        k_act = max(0, min(classifier.num_layers, int(action[0])))
        b_idx = max(0, min(3, int(action[1])))
        bw_val = INDEX_TO_BITWIDTH[b_idx]

        # 1. Real PyTorch Split Inference on the text message
        nlp_res = classifier.predict_split(text=text, split_point=k_act, bitwidth=bw_val)

        # 2. SimPy DES + MEC Environment simulation step
        obs, reward, term, trunc, info = env.step(action)
        m = info["metrics"]

        y_true.append(int(label))
        y_pred.append(int(nlp_res.predicted_class))
        if nlp_res.top1_match_vs_fp32:
            agreements += 1

        lat_ms = float(m["t_total"] * 1000.0)
        e_mj = float(m["e_mobile"] * 1000.0)
        latencies_ms.append(lat_ms)
        energies_mj.append(e_mj)
        tx_bytes_list.append(int(nlp_res.transmitted_bytes))
        split_points.append(k_act)
        bitwidths.append(nlp_res.bitwidth)
        if m["sla_satisfied"]:
            sla_hits += 1

        if idx < 25:
            sample_logs.append({
                "Message": text[:70] + ("..." if len(text) > 70 else ""),
                "True": "SPAM" if label == 1 else "HAM",
                "Predicted": nlp_res.predicted_label,
                "Prob(Spam)": nlp_res.spam_probability,
                "Tokens": nlp_res.num_tokens,
                "Split_k": k_act,
                "Bits": f"INT{nlp_res.bitwidth}" if nlp_res.bitwidth < 32 else "FP32",
                "Tx_KB": round(nlp_res.transmitted_bytes / 1024.0, 2),
                "Latency_ms": round(lat_ms, 1),
                "Energy_mJ": round(e_mj, 2)
            })

        if term or trunc:
            # Continue evaluating full test set cleanly while preserving SimPy queues
            env.current_step = 0

    cls_metrics = compute_classification_metrics(y_true, y_pred)
    sorted_lat = sorted(latencies_ms)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    return {
        "algorithm": policy.name,
        "scenario": env.network.config.scenario.value.capitalize(),
        "accuracy_pct": round(cls_metrics["accuracy"] * 100.0, 2),
        "precision_pct": round(cls_metrics["precision"] * 100.0, 2),
        "recall_pct": round(cls_metrics["recall"] * 100.0, 2),
        "f1_pct": round(cls_metrics["f1_score"] * 100.0, 2),
        "fp32_agreement_pct": round((agreements / len(test_texts)) * 100.0, 2),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "p95_latency_ms": round(sorted_lat[p95_idx], 2),
        "avg_energy_mj": round(sum(energies_mj) / len(energies_mj), 2),
        "total_tx_kb": round(sum(tx_bytes_list) / 1024.0, 2),
        "sla_success_pct": round((sla_hits / len(test_texts)) * 100.0, 1),
        "avg_split_k": round(sum(split_points) / len(split_points), 2),
        "avg_bitwidth": round(sum(bitwidths) / len(bitwidths), 1),
        "confusion_matrix": {
            "tp": cls_metrics["tp"],
            "fp": cls_metrics["fp"],
            "tn": cls_metrics["tn"],
            "fn": cls_metrics["fn"]
        },
        "sample_logs": sample_logs
    }
