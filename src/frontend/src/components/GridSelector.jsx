import { useEffect, useRef, useState } from "react";
import { IoIosArrowDown } from "react-icons/io";

const DEFAULT_OPTIONS = [
  { code: "program", name: "Programm" },
  { code: "kommunalomat", name: "Kommunalomat" },
];

const GridSelector = ({
  options = DEFAULT_OPTIONS,
  selected = "kommunalomat",
  onSelect,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const current = options.find((o) => o.code === selected) || options[0];

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleDropdown = () => setIsOpen((s) => !s);

  const handleSelect = (code) => {
    if (onSelect) onSelect(code);
    setIsOpen(false);
  };

  return (
    <div
      className="relative max-w-xs mx-auto mr-8 font-sans"
      ref={dropdownRef}
    >
      <button
        onClick={toggleDropdown}
        className={`
          w-full xs:w-44
          rounded-full
          font-semibold
          text-md
          flex items-center justify-between sm:justify-between
          bg-secondary text-white
          px-4 py-2
          transition-all duration-300 ease-in-out
          focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-50
        `}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span>{current.name}</span>
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
            max-h-60 overflow-y-auto
          `}
          role="listbox"
        >
          {options.map((opt) => (
            <button
              key={opt.code}
              onClick={() => handleSelect(opt.code)}
              className={`
                block w-full text-left
                px-4 py-2 h-10
                gap-2
                bg-white
                text-gray-800 hover:bg-gray-100
                font-medium
                transition-colors duration-200
                ${selected === opt.code ? "bg-gray-100 text-secondary" : ""}
              `}
              role="option"
              aria-selected={selected === opt.code}
            >
              {opt.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default GridSelector;
