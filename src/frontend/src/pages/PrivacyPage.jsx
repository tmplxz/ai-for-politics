import { useTranslation } from "react-i18next";

function PrivacyPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center justify-center px-4 md:px-0">
      <div className="w-full max-w-[50%]">
        <h1 className="text-primary py-8 md:p-6 text-2xl md:text-3xl font-bold text-center">
          {t("priv.name")}
        </h1>
        <div
          dangerouslySetInnerHTML={{ __html: t("priv.content") }}
        />
      </div>
    </div>
  );
}

export default PrivacyPage;