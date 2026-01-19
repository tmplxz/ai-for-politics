import { useEffect, useRef, useState } from "react";
import { IoIosArrowDown } from "react-icons/io";

const LanguageSelector = ({ languages, selectedLanguage, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const currentLangObj =
    languages.find((lang) => lang.code === selectedLanguage) || languages[0];

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  const handleLanguageClick = (langCode) => {
    onSelect(langCode);
    setIsOpen(false);
  };

  return (
    <div
      className="relative w-full max-w-xs mx-auto my-2 font-sans"
      ref={dropdownRef}
    >
      <button
        onClick={toggleDropdown}
        className={`
          w-24 sm:w-36
          rounded-full
          font-semibold
          text-md
          flex items-center justify-between
          bg-secondary text-white
          px-4 py-2
          transition-all duration-300 ease-in-out
          focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-50
        `}
      >
        {/* Show code on mobile, full name on desktop */}
        <span>
          <span className="sm:hidden">{currentLangObj.code.toUpperCase()}</span>
          <span className="hidden sm:inline">{currentLangObj.name}</span>
        </span>

        <IoIosArrowDown
          className={`w-5 h-5 ml-2 transition-transform duration-300 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <div
          className={`
            absolute top-full left-0 right-0
            mt-2 p-2
            bg-white rounded-xl shadow-lg
            z-10
            origin-top duration-300 ease-out
            ${isOpen ? "scale-y-100 opacity-100" : "scale-y-0 opacity-0"}
            max-h-60 overflow-y-auto
          `}
        >
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => handleLanguageClick(lang.code)}
              className={`
                block w-full text-left
                px-4 py-2 h-10
                gap-2
                bg-white
                text-gray-800 hover:bg-gray-100
                font-medium
                transition-colors duration-200
                ${
                  selectedLanguage === lang.code
                    ? "bg-gray-100 text-secondary"
                    : ""
                }
              `}
            >
              {/* Show code on mobile, full name on desktop */}
              <span className="sm:hidden">{lang.code.toUpperCase()}</span>
              <span className="hidden sm:inline">{lang.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageSelector;
