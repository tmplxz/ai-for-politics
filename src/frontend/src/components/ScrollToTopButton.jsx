import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { IoIosArrowUp } from "react-icons/io";

const ScrollToTopButton = () => {
  const [showButton, setShowButton] = useState(false);
  const { t } = useTranslation();

  useEffect(() => {
    const handleScroll = () => {
      const scrolled =
        window.scrollY / (document.body.scrollHeight - window.innerHeight);
      setShowButton(scrolled > 0.1);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <button
      onClick={scrollToTop}
      className={`fixed bottom-6 h-14 w-14 right-6 z-50 rounded-full bg-primary text-white shadow-lg hover:bg-secondary transition-all duration-300 transform ${
        showButton
          ? "opacity-100 scale-100"
          : "opacity-0 scale-90 pointer-events-none"
      }`}
      title={t("scroll_to_top_button")}
    >
      <IoIosArrowUp />
    </button>
  );
};

export default ScrollToTopButton;
