"""Local Hebrew-to-English translation using a one-time Argos model download."""

from __future__ import annotations


class TranslationUnavailable(RuntimeError):
    """Raised when the optional local translation runtime is not ready."""


def _argos_modules():
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError as error:
        raise TranslationUnavailable(
            "Local translation is not installed. Run: py -m pip install -r requirements.txt "
            "then py web_app.py --install-hebrew-translation-model"
        ) from error
    return argostranslate.package, argostranslate.translate


def _has_hebrew_to_english_model(package_module: object) -> bool:
    return any(
        package.from_code == "he" and package.to_code == "en"
        for package in package_module.get_installed_packages()
    )


def _install_hebrew_to_english_model(package_module: object) -> None:
    """Download and install the Hebrew-to-English Argos model."""

    package_module.update_package_index()
    available_packages = package_module.get_available_packages()
    try:
        hebrew_to_english = next(
            package
            for package in available_packages
            if package.from_code == "he" and package.to_code == "en"
        )
    except StopIteration as error:
        raise TranslationUnavailable("The Hebrew-to-English translation model is unavailable.") from error
    hebrew_to_english.install()


def install_hebrew_to_english_model() -> None:
    """Download the one required model once, after Argos Translate is installed."""

    package_module, _ = _argos_modules()
    if not _has_hebrew_to_english_model(package_module):
        _install_hebrew_to_english_model(package_module)


def prepare_hebrew_to_english_translation() -> None:
    """Install, if needed, and warm the local translator during server startup."""

    package_module, translate_module = _argos_modules()
    if not _has_hebrew_to_english_model(package_module):
        _install_hebrew_to_english_model(package_module)
    try:
        # A short local translation loads the installed model before the first
        # user query, avoiding a noticeable first-search delay.
        translate_module.translate("שלום", "he", "en")
    except Exception as error:
        raise TranslationUnavailable("The local Hebrew translator could not be prepared.") from error


def translate_hebrew_to_english(text: str) -> str:
    """Translate a Hebrew transcript locally after the model was installed."""

    package_module, translate_module = _argos_modules()
    if not _has_hebrew_to_english_model(package_module):
        raise TranslationUnavailable(
            "The local Hebrew-to-English model is not installed. Run: "
            "py web_app.py --install-hebrew-translation-model"
        )
    translated = translate_module.translate(text, "he", "en").strip()
    if not translated:
        raise TranslationUnavailable("The local translator did not return text. Try again.")
    return translated
