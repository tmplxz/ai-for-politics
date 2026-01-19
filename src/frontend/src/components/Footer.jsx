import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

const base = import.meta.env.BASE_URL;

const Footer = () => {
  const { t } = useTranslation();
  return (
    <footer className="bg-white rounded-lg shadow-sm m-4 w-full">
      <div
        className="w-full p-4 md:py-8"
        style={{ paddingLeft: "10%", paddingRight: "10%" }}
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between items-center text-center">
          <a
            href="https://github.com/tmplxz/ai-for-politics"
            className="flex items-center mb-4 sm:mb-0 space-x-3 rtl:space-x-reverse"
          >
            <span className="text-black self-center text-xl whitespace-nowrap dark:text-white">
              <p>Authors</p>
              <p>(anonymized)</p>
            </span>
          </a>
            <ul className="flex flex-row items-center mb-6 text-sm font-medium text-gray-500 dark:text-gray-400 p-4">
              <li>
                <Link
                  to="/impressum"
                  className="text-primary hover:underline p-4"
                >
                  {t("impr.name")}
                </Link>
              </li>
              <li>
                <Link
                  to="/datenschutz"
                  className="text-primary hover:underline p-4"
                >
                  {t("priv.name")}
                </Link>
              </li>
            </ul>
        </div>
        <hr className="my-6 border-gray-200 sm:mx-auto dark:border-gray-700 lg:my-8" />
        <span className="block text-sm text-gray-500 sm:text-center dark:text-gray-400 text-center">
          © 2025{" "}
          <a
            href="https://github.com/tmplxz/ai-for-politics"
            className="text-primary hover:underline hover:text-black"
          >
            Authors (anonymized)
          </a>
          . All Rights Reserved.
        </span>
      </div>
    </footer>
  );
};

export default Footer;
