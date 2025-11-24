import torch

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


LANGUAGE_MAPPING: dict[str, str] = {
    'ar': 'ar_AR',
    'cs': 'cs_CZ',
    'de': 'de_DE',
    'en': 'en_XX',
    'es': 'es_XX',
    'et': 'et_EE',
    'fi': 'fi_FI',
    'fr': 'fr_XX',
    'gu': 'gu_IN',
    'hi': 'hi_IN',
    'it': 'it_IT',
    'ja': 'ja_XX',
    'kk': 'kk_KZ',
    'ko': 'ko_KR',
    'lt': 'lt_LT',
    'lv': 'lv_LV',
    'my': 'my_MM',
    'ne': 'ne_NP',
    'nl': 'nl_XX',
    'ro': 'ro_RO',
    'ru': 'ru_RU',
    'si': 'si_LK',
    'tr': 'tr_TR',
    'vi': 'vi_VN',
    'zh': 'zh_CN',
    'af': 'af_ZA',
    'az': 'az_AZ',
    'bn': 'bn_IN',
    'fa': 'fa_IR',
    'he': 'he_IL',
    'hr': 'hr_HR',
    'id': 'id_ID',
    'ka': 'ka_GE',
    'km': 'km_KH',
    'mk': 'mk_MK',
    'ml': 'ml_IN',
    'mn': 'mn_MN',
    'mr': 'mr_IN',
    'pl': 'pl_PL',
    'ps': 'ps_AF',
    'pt': 'pt_XX',
    'sv': 'sv_SE',
    'sw': 'sw_KE',
    'ta': 'ta_IN',
    'te': 'te_IN',
    'th': 'th_TH',
    'tl': 'tl_XX',
    'uk': 'uk_UA',
    'ur': 'ur_PK',
    'xh': 'xh_ZA',
    'gl': 'gl_ES',
    'sl': 'sl_SI',
}


class MbartTranslate:
    '''
    A set of functions used to augment text.
    Supported languages are:
    Language Name	Code

    Arabic          ar
    Czech           cs
    German          de
    English         en
    Spanish         es
    Estonian        et
    Finnish         fi
    French          fr
    Gujarati        gu
    Hindi           hi
    Italian         it
    Japanese        ja
    Kazakh          kk
    Korean          ko
    Lithuanian      lt
    Latvian         lv
    Burmese         my
    Nepali          ne
    Dutch           nl
    Romanian        ro
    Russian         ru
    Sinhala         si
    Turkish         tr
    Vietnamese      vi
    Chinese         zh
    Afrikaans       af
    Azerbaijani     az
    Bengali         bn
    Persian         fa
    Hebrew          he
    Croatian        hr
    Indonesian      id
    Georgian        ka
    Khmer           km
    Macedonian      mk
    Malayalam       ml
    Mongolian       mn
    Marathi         mr
    Polish          pl
    Pashto          ps
    Portuguese      pt
    Swedish         sv
    Swahili         sw
    Tamil           ta
    Telugu          te
    Thai            th
    Tagalog         tl
    Ukrainian       uk
    Urdu            ur
    Xhosa           xh
    Galician        gl
    Slovene         sl

    Example usage: ::
        >>> from textaugment import MbartTranslate
        >>> t = MbartTranslate(src='en',to='es')
        >>> t.augment('I love school')
        i adore school
    '''

    def __init__(
        self,
        src: str, 
        to: str,
        model_name: str = 'facebook/mbart-large-50-many-to-many-mmt'
    ) -> None:
        '''
        A method to initialize parameters

        :type src:      str
        :param src:     source language of the text
        :param to:      str
        :param to:      Destination language to translate to. The language should be a family of the source language for better results. 
                        The text will then be translated back to the source language
        :rtype:         None
        :return:        constructer do not return
        '''
        if src not in LANGUAGE_MAPPING or to not in LANGUAGE_MAPPING:
            raise KeyError('src or to language not supported.')
        self._from_locale: str | None = LANGUAGE_MAPPING.get(src)
        self._to_locale: str | None = LANGUAGE_MAPPING.get(to)

        self._model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(model_name)
        self._tokenizer: MBart50TokenizerFast = MBart50TokenizerFast.from_pretrained(model_name)

    def _translate(self, text: str, from_locale: str, to_locale: str) -> str:
        '''
        A private method to translate from a src_locale to tgt_locale using Mbart50-many-to-many

        :type text:         str
        :param text:        text to be translated
        :type text:         str
        :param from_locale: source language locale
        :type text:         str
        :param to_locale:   target language locale
        :rtype text:         str
        :return:            the translated text
        '''
        self._tokenizer.src_lang = from_locale
        encoded_text: BatchEncoding = self._tokenizer(text, return_tensors='pt')
        generated_tokens: torch.Tensor = self._model.generate(
            **encoded_text,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[to_locale]
        )
        return self._tokenizer.decode(generated_tokens.squeeze(0), skip_special_tokens=True)

    def augment(self, text: str) -> str:
        '''
        A method to paraphrase a sentence.
        
        :type text:     str
        :param text:    sentence used for text augmentation
        :rtype:         str
        :return:        The augmented text
        '''
        translated_text: str = self._translate(text.lower(), self._from_locale, self._to_locale)
        return self._translate(translated_text, self._to_locale, self._from_locale)
