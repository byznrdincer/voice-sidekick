import re


class TTSTextProcessor:
    """
    UI'daki cevabı değiştirmeden, yalnızca TTS motoruna gidecek metni
    konuşmaya uygun düz metne dönüştürür.
    """

    def prepare(self, text: str) -> str:
        if not text:
            return ""

        prepared = text.strip()

        prepared = self._markdown_links_to_labels(prepared)
        prepared = self._remove_raw_urls(prepared)
        prepared = self._normalize_emails(prepared)
        prepared = self._remove_markdown(prepared)
        prepared = self._remove_list_markers(prepared)
        prepared = self._normalize_currency(prepared)
        prepared = self._normalize_percentages(prepared)
        prepared = self._normalize_times(prepared)
        prepared = self._normalize_numbers(prepared)
        prepared = self._normalize_punctuation(prepared)
        prepared = self._normalize_whitespace(prepared)

        return prepared

    def split(self, text: str, max_chars: int = 400) -> list[str]:
        """
        TTS metnini mümkün olduğunca cümle sınırlarından böler.
        Çok uzun tek cümleleri kelime sınırından parçalar.
        """
        text = (text or "").strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            pieces = self._split_long_sentence(
                sentence,
                max_chars,
            )

            for piece in pieces:
                candidate = f"{current} {piece}".strip()

                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate

        if current:
            chunks.append(current)

        return chunks

    def _split_long_sentence(
        self,
        sentence: str,
        max_chars: int,
    ) -> list[str]:

        if len(sentence) <= max_chars:
            return [sentence]

        pieces = []
        current = ""

        for word in sentence.split():
            candidate = f"{current} {word}".strip()

            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = word
            else:
                current = candidate

        if current:
            pieces.append(current)

        return pieces

    def _markdown_links_to_labels(self, text: str) -> str:
        # [GitHub](https://github.com/...) -> GitHub
        return re.sub(
            r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)",
            r"\1",
            text,
        )

    def _remove_raw_urls(self, text: str) -> str:
        return re.sub(
            r"https?://\S+|www\.\S+",
            "",
            text,
        )

    def _normalize_emails(self, text: str) -> str:
        pattern = (
            r"\b[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        def replace_email(match):
            email = match.group(0)

            email = email.replace("@", " et ")
            email = email.replace(".", " nokta ")
            email = email.replace("_", " alt çizgi ")
            email = email.replace("-", " tire ")

            return email

        return re.sub(
            pattern,
            replace_email,
            text,
        )

    def _remove_markdown(self, text: str) -> str:

        # ### Başlık -> Başlık.
        text = re.sub(
            r"(?m)^\s*#{1,6}\s*(.+?)\s*$",
            lambda m: m.group(1).rstrip(".:") + ".",
            text,
        )

        # Bold / italic
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"~~(.*?)~~", r"\1", text)
        text = re.sub(
            r"(?<!\*)\*([^*\n]+)\*(?!\*)",
            r"\1",
            text,
        )
        text = re.sub(
            r"(?<!_)_([^_\n]+)_(?!_)",
            r"\1",
            text,
        )

        # Inline code
        text = re.sub(
            r"`([^`]+)`",
            r"\1",
            text,
        )

        # Blockquote
        text = re.sub(
            r"(?m)^\s*>\s?",
            "",
            text,
        )

        return text

    def _remove_list_markers(self, text: str) -> str:

        # 1. Görev / 1) Görev
        text = re.sub(
            r"(?m)^\s*\d+[.)]\s+",
            "",
            text,
        )

        # -, *, +, •
        text = re.sub(
            r"(?m)^\s*[-*+•]\s+",
            "",
            text,
        )

        return text

    def _normalize_currency(self, text: str) -> str:
        return re.sub(
            r"\b(\d+)\s*(?:TL|₺)\b",
            lambda m: (
                self._number_to_turkish(
                    int(m.group(1))
                )
                + " Türk lirası"
            ),
            text,
            flags=re.IGNORECASE,
        )

    def _normalize_percentages(self, text: str) -> str:
        return re.sub(
            r"%\s*(\d+)",
            lambda m: (
                "yüzde "
                + self._number_to_turkish(
                    int(m.group(1))
                )
            ),
            text,
        )

    def _normalize_times(self, text: str) -> str:
        return re.sub(
            r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b",
            lambda m: (
                self._number_to_turkish(
                    int(m.group(1))
                )
                + " "
                + self._number_to_turkish(
                    int(m.group(2))
                )
            ),
            text,
        )

    def _normalize_numbers(self, text: str) -> str:
        return re.sub(
            r"\b\d+\b",
            lambda m: self._number_to_turkish(
                int(m.group(0))
            ),
            text,
        )

    def _normalize_punctuation(self, text: str) -> str:

        # Etiket sonları:
        # Saat: 14.30 -> Saat. on dört otuz
        text = re.sub(
            r":\s*",
            ". ",
            text,
        )

        text = re.sub(
            r"[.]{2,}",
            ".",
            text,
        )

        text = re.sub(
            r"[!?]{2,}",
            lambda m: m.group(0)[0],
            text,
        )

        return text

    def _normalize_whitespace(self, text: str) -> str:

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Satır sonlarını doğal duraklara çevir.
        text = re.sub(
            r"\s*\n\s*",
            ". ",
            text,
        )

        text = re.sub(
            r"(?:\.\s*){2,}",
            ". ",
            text,
        )

        text = text.strip(" .")

        if text:
            text += "."

        return text

    def _number_to_turkish(self, n: int) -> str:

        ones = [
            "",
            "bir",
            "iki",
            "üç",
            "dört",
            "beş",
            "altı",
            "yedi",
            "sekiz",
            "dokuz",
        ]

        tens = [
            "",
            "on",
            "yirmi",
            "otuz",
            "kırk",
            "elli",
            "altmış",
            "yetmiş",
            "seksen",
            "doksan",
        ]

        if n == 0:
            return "sıfır"

        if n < 0:
            return (
                "eksi "
                + self._number_to_turkish(-n)
            )

        def under_thousand(x: int) -> str:
            parts = []

            hundreds = x // 100
            rest = x % 100

            if hundreds:
                parts.append(
                    "yüz"
                    if hundreds == 1
                    else ones[hundreds] + " yüz"
                )

            if rest >= 10:
                parts.append(
                    tens[rest // 10]
                )

            if rest % 10:
                parts.append(
                    ones[rest % 10]
                )

            return " ".join(parts)

        groups = [
            (1_000_000_000, "milyar"),
            (1_000_000, "milyon"),
            (1_000, "bin"),
        ]

        parts = []
        remainder = n

        for value, name in groups:
            amount = remainder // value

            if amount:
                if value == 1_000 and amount == 1:
                    parts.append("bin")
                else:
                    parts.append(
                        self._number_to_turkish(amount)
                        + " "
                        + name
                    )

                remainder %= value

        if remainder:
            parts.append(
                under_thousand(remainder)
            )

        return " ".join(parts)