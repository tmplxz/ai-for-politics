import { useTranslation } from "react-i18next";

const Disclaimer = () => {
  const { t } = useTranslation();

  return (
    <div className="border-4 border-secondary my-10 p-4">
      <p className="text-secondary font-bold text-xl">
        {t("disclaimer.title")}
      </p>
      <br></br>
      <div dangerouslySetInnerHTML={{ __html: t("disclaimer.text") }} />
    </div>
  );
};

export default Disclaimer;