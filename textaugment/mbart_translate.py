import torch

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


LANGUAGE_MAPPING: dict[str, dict[str, str]] = {
    'ar': {'locale': 'ar_AR', 'name': 'Arabic'},
    'cs': {'locale': 'cs_CZ', 'name': 'Czech'},
    'de': {'locale': 'de_DE', 'name': 'German'},
    'en': {'locale': 'en_XX', 'name': 'English'},
    'es': {'locale': 'es_XX', 'name': 'Spanish'},
    'et': {'locale': 'et_EE', 'name': 'Estonian'},
    'fi': {'locale': 'fi_FI', 'name': 'Finnish'},
    'fr': {'locale': 'fr_XX', 'name': 'French'},
    'gu': {'locale': 'gu_IN', 'name': 'Gujarati'},
    'hi': {'locale': 'hi_IN', 'name': 'Hindi'},
    'it': {'locale': 'it_IT', 'name': 'Italian'},
    'ja': {'locale': 'ja_XX', 'name': 'Japanese'},
    'kk': {'locale': 'kk_KZ', 'name': 'Kazakh'},
    'ko': {'locale': 'ko_KR', 'name': 'Korean'},
    'lt': {'locale': 'lt_LT', 'name': 'Lithuanian'},
    'lv': {'locale': 'lv_LV', 'name': 'Latvian'},
    'my': {'locale': 'my_MM', 'name': 'Burmese'},
    'ne': {'locale': 'ne_NP', 'name': 'Nepali'},
    'nl': {'locale': 'nl_XX', 'name': 'Dutch'},
    'ro': {'locale': 'ro_RO', 'name': 'Romanian'},
    'ru': {'locale': 'ru_RU', 'name': 'Russian'},
    'si': {'locale': 'si_LK', 'name': 'Sinhala'},
    'tr': {'locale': 'tr_TR', 'name': 'Turkish'},
    'vi': {'locale': 'vi_VN', 'name': 'Vietnamese'},
    'zh': {'locale': 'zh_CN', 'name': 'Chinese'},
    'af': {'locale': 'af_ZA', 'name': 'Afrikaans'},
    'az': {'locale': 'az_AZ', 'name': 'Azerbaijani'},
    'bn': {'locale': 'bn_IN', 'name': 'Bengali'},
    'fa': {'locale': 'fa_IR', 'name': 'Persian'},
    'he': {'locale': 'he_IL', 'name': 'Hebrew'},
    'hr': {'locale': 'hr_HR', 'name': 'Croatian'},
    'id': {'locale': 'id_ID', 'name': 'Indonesian'},
    'ka': {'locale': 'ka_GE', 'name': 'Georgian'},
    'km': {'locale': 'km_KH', 'name': 'Khmer'},
    'mk': {'locale': 'mk_MK', 'name': 'Macedonian'},
    'ml': {'locale': 'ml_IN', 'name': 'Malayalam'},
    'mn': {'locale': 'mn_MN', 'name': 'Mongolian'},
    'mr': {'locale': 'mr_IN', 'name': 'Marathi'},
    'pl': {'locale': 'pl_PL', 'name': 'Polish'},
    'ps': {'locale': 'ps_AF', 'name': 'Pashto'},
    'pt': {'locale': 'pt_XX', 'name': 'Portuguese'},
    'sv': {'locale': 'sv_SE', 'name': 'Swedish'},
    'sw': {'locale': 'sw_KE', 'name': 'Swahili'},
    'ta': {'locale': 'ta_IN', 'name': 'Tamil'},
    'te': {'locale': 'te_IN', 'name': 'Telugu'},
    'th': {'locale': 'th_TH', 'name': 'Thai'},
    'tl': {'locale': 'tl_XX', 'name': 'Tagalog'},
    'uk': {'locale': 'uk_UA', 'name': 'Ukrainian'},
    'ur': {'locale': 'ur_PK', 'name': 'Urdu'},
    'xh': {'locale': 'xh_ZA', 'name': 'Xhosa'},
    'gl': {'locale': 'gl_ES', 'name': 'Galician'},
    'sl': {'locale': 'sl_SI', 'name': 'Slovene'}
}


class MbartTranslate:
    def __init__(self, src: str, to: str) -> None:
        if src not in LANGUAGE_MAPPING or to not in LANGUAGE_MAPPING:
            raise KeyError('src or to language not supported.')
        self._from_locale: str = LANGUAGE_MAPPING.get(src, {}).get('locale', '')
        self._to_locale: str = LANGUAGE_MAPPING.get(to, {}).get('locale', '')

        model_name: str = 'facebook/mbart-large-50-many-to-many-mmt'
        self._model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(model_name)
        self._tokenizer: MBart50TokenizerFast = MBart50TokenizerFast.from_pretrained(model_name)

    def _translate(self, text: str, from_locale: str, to_locale: str) -> str:
        self._tokenizer.src_lang = from_locale
        encoded_text: BatchEncoding = self._tokenizer(text, return_tensors='pt')
        generated_tokens: torch.Tensor = self._model.generate(
            **encoded_text,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[to_locale]
        )
        return self._tokenizer.decode(generated_tokens.squeeze(0), skip_special_tokens=True)

    def augment(self, text: str) -> str:
        translated_text: str = self._translate(text.lower(), self._from_locale, self._to_locale)
        return self._translate(translated_text, self._to_locale, self._from_locale)
