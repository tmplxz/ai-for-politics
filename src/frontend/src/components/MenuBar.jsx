import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { FiMenu, FiX } from "react-icons/fi";
import LanguageSelector from "./LanguageSelector";

const base = import.meta.env.BASE_URL;

export default function MenuBar() {
  const { t, i18n } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);

  const switchLanguage = (lng) => {
    i18n.changeLanguage(lng);
  };

  const languages = [
    { name: "Deutsch", code: "de" },
    { name: "English", code: "en" },
  ];

  return (
    <div className="w-full flex flex-col justify-between items-center mb-4 p-2 rounded">
      <div className="flex w-full items-center justify-between flex-col xl:flex-row">
        {/* Logos */}
        <div className="flex flex-row items-center justify-center mb-2 sm:mb-0">
          <a href="https://github.com/tmplxz/ai-for-politics" target="_blank">
            <img
              src={`${base}logo.png`}
              alt="Logo"
              className="h-20 w-20 sm:h-28 sm:w-28 mr-2 sm:mr-4 ml-2 sm:ml-8"
            />
          </a>
        </div>

        {/* Headline */}
        <h1 className="cursor-pointer text-3xl sm:text-5xl font-bold text-secondary text-center xl:absolute xl:left-1/2 xl:-translate-x-1/2 mb-2 sm:mb-0">
          <Link
            to="/"
            className="text-inherit no-underline hover:no-underline focus:outline-none hover:text-secondary"
          >
            {t("title")}
          </Link>
        </h1>

        {/* Language selector + hamburger */}
        <div className="flex gap-2 mt-2 sm:mr-8">
          <LanguageSelector
            languages={languages}
            selectedLanguage={i18n.language}
            onSelect={switchLanguage}
          />
          <div className="sm:hidden flex items-center text-white p-2">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="bg-secondary w-24 h-10 rounded-full flex items-center justify-center"
            >
              {menuOpen ? <FiX size={28} /> : <FiMenu size={28} />}
            </button>
          </div>
        </div>
      </div>

      {/* Desktop nav */}
      <nav className="hidden sm:flex flex-col sm:flex-row gap-3 sm:gap-5 text-black justify-center w-full mt-2 sm:mt-0">
        <div className="cursor-pointer font-bold decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
          <Link to="/" style={{ color: "inherit", textDecoration: "none" }}>
            {t("nav_bar.home")}
          </Link>
        </div>
        <div className="cursor-pointer font-bold decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
          <Link
            to="/projekt"
            style={{ color: "inherit", textDecoration: "none" }}
          >
            {t("nav_bar.projekt")}
          </Link>
        </div>
        <div className="cursor-pointer font-bold decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
          <Link
            to="/technik"
            style={{ color: "inherit", textDecoration: "none" }}
          >
            {t("nav_bar.technik")}
          </Link>
        </div>
      </nav>

      {/* Mobile dropdown nav */}
      {menuOpen && (
        <nav className="flex sm:hidden flex-col gap-3 text-black justify-center w-full mt-2 font-bold">
          <div className="cursor-pointer decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
            <Link to="/" style={{ color: "inherit", textDecoration: "none" }} onClick={() => setMenuOpen(false)}>
              {t("nav_bar.home")}
            </Link>
          </div>
          <div className="cursor-pointer decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
            <Link to="/projekt" style={{ color: "inherit", textDecoration: "none" }} onClick={() => setMenuOpen(false)}>
              {t("nav_bar.projekt")}
            </Link>
          </div>
          <div className="cursor-pointer decoration-primary underline decoration-2 decoration-offset-2 hover:text-secondary">
            <Link to="/technik" style={{ color: "inherit", textDecoration: "none" }} onClick={() => setMenuOpen(false)}>
              {t("nav_bar.technik")}
            </Link>
          </div>
        </nav>
      )}
    </div>
  );
}
