#!/usr/bin/env python
# WordNet-based data augmentation 
#
# Copyright (C) 2020-2026
# Author: Joseph Sefara
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE
import asyncio
# from asyncio import AbstractEventLoop
import nest_asyncio
from .constants import LANGUAGES
from googletrans import Translator
from googletrans.models import Translated


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
        LANGUAGE_CODES: list[str] = list(LANGUAGES.keys())

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

    def __event_loop_running(self) -> bool:
        '''
        checks whether an asyncio event loop is already running.

        :rtype:         bool
        :return:        True or False
        '''
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def __in_jupyter(self) -> bool:
        '''
        checks whether Translate is running inside Jupyter Notebook / Lab / Colab.

        :rtype:         bool
        :return:        True or False
        '''
        try:
            
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'
        except Exception:
            return False

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
                
        if not self.__event_loop_running():
            return asyncio.run(translate_text())

        if self.__in_jupyter():
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(translate_text())

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(translate_text())
