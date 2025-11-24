#!/usr/bin/env python
# WordNet-based data augmentation 
#
# Copyright (C) 2020
# Author: Joseph Sefara
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE
import asyncio
from asyncio import AbstractEventLoop

from googletrans import Translator
from googletrans.models import Translated


LANGUAGE_CODES: list[str] = [
    'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 
    'bn', 'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-cn', 'zh-tw', 
    'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'tl', 
    'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 
    'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 
    'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'km',
    'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb',
    'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 
    'my', 'ne', 'no', 'ps', 'fa', 'pl', 'pt', 'pa', 
    'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 
    'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 
    'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'uz', 
    'vi', 'cy', 'xh', 'yi', 'yo', 'zu', 'fil', 'he'
]


class Translate: 
    '''
    A set of functions used to augment data.
    Supported languages are:
    Language Name	    Code

    Afrikaans           af
    Albanian            sq
    Arabic	            ar
    Azerbaijani         az
    Basque              eu
    Bengali             bn
    Belarusian	        be
    Bulgarian	        bg
    Catalan	            ca
    Chinese Simplified	zh-CN
    Chinese Traditional	zh-TW
    Croatian	        hr
    Czech	            cs
    Danish	            da
    Dutch	            nl
    English	            en
    Esperanto	        eo
    Estonian	        et
    Filipino	        tl
    Finnish	            fi
    French	            fr
    Galician	        gl
    Georgian	        ka
    German	            de
    Greek	            el
    Gujarati	        gu
    Haitian Creole	    ht
    Hebrew	            iw
    Hindi	            hi
    Hungarian	        hu
    Icelandic	        is
    Indonesian	        id
    Irish	            ga
    Italian	            it
    Japanese	        ja
    Kannada	            kn
    Korean	            ko
    Latin	            la
    Latvian	            lv
    Lithuanian	        lt
    Macedonian	        mk
    Malay	            ms
    Maltese	            mt
    Norwegian	        no
    Persian	            fa
    Polish	            pl
    Portuguese	        pt
    Romanian	        ro
    Russian	            ru
    Serbian	            sr
    Slovak	            sk
    Slovenian	        sl
    Spanish	            es
    Swahili	            sw
    Swedish	            sv
    Tamil	            ta
    Telugu	            te
    Thai	            th
    Turkish	            tr
    Ukrainian	        uk
    Urdu	            ur
    Vietnamese	        vi
    Welsh	            cy
    Yiddish	            yi

    Example usage: ::
        >>> from textaugment import Translate
        >>> t = Translate(src='en',to='es')
        >>> t.augment('I love school')
        i adore school
    '''
    
    def __init__(self, **kwargs):
        '''
        A method to initialize parameters

        :type src:      str
        :param src:     source language of the text
        :type to:       str
        :param to:      Destination language to translate to. The language should be a family of the source language for better results
                        The text will then be translated back to the source language
        :rtype:         None
        :return:        constructer do not return
        '''
        try:
            if 'to' not in kwargs:
                raise ValueError("'to' missing")
            elif 'src' not in kwargs:
                raise ValueError("'src' missing")
            if kwargs['to'] not in LANGUAGE_CODES:
                raise KeyError('Value of to is not surpported. See help(Translate)')
            if kwargs['src'] not in LANGUAGE_CODES:
                raise KeyError('Value of src is not surpported. See help(Translate)')
        except (ValueError, KeyError):
            print("The values of the keys 'to' and 'src' are required. E.g Translate(src='en', to='es')")
            raise
        else:    
            self.to = kwargs['to']
            self.src = kwargs['src']

    def augment(self, text: str) -> str:
        '''
        A method to paraphrase a sentence.
        
        :type text:     str
        :param text:    sentence used for text augmentation 
        :rtype:         str
        :return:        the augmented text
        '''
        if type(text) is not str:
            raise TypeError('DataType must be a string')
                
        async def translate_text() -> str:
            async with Translator() as translator:
                forward: Translated = await translator.translate(text.lower(), dest=self.to, src=self.src)
                backward: Translated = await translator.translate(forward.text, dest=self.src, src=self.to)
                return backward.text
                
        event_loop: AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        back_translated_text: str = event_loop.run_until_complete(translate_text())
        event_loop.close()

        return back_translated_text
