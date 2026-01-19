import Disclaimer from "../components/Disclaimer";
import EmissionsTable from "../components/EmissionsTable";
import { useTranslation } from "react-i18next";

const base = import.meta.env.BASE_URL;

export default function TechPage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center justify-center px-4 md:px-0">
        <h1 className="justify-self-center text-secondary py-6 md:py-8 text-center whitespace-normal md:whitespace-nowrap break-words">
          {t("technik.heading_tech")}
        </h1>
      <div
        className="w-full lg:max-w-[50%] break-words"
        dangerouslySetInnerHTML={{ __html: t("technik.text_tech1") }}
      />
      <div className="w-full lg:max-w-[50%] my-8">
        <img
          src={`${base}pipeline_en.png`}
          className="w-full rounded"
          alt="pipeline"
        />
      </div>
      <div
        className="w-full lg:max-w-[50%] break-words"
        dangerouslySetInnerHTML={{ __html: t("technik.text_tech2") }}
      />
      {/* Deployment and Emissions */}
      <h1 className="justify-self-center text-secondary py-6 md:py-8 text-center whitespace-normal md:whitespace-nowrap break-words">
        {t("technik.heading_deploy")}
      </h1>
      <div
        className="w-full lg:max-w-[50%] break-words"
        dangerouslySetInnerHTML={{ __html: t("technik.text_deploy1") }}
      />
      <EmissionsTable />
      <div
        className="w-full lg:max-w-[50%] break-words"
        dangerouslySetInnerHTML={{ __html: t("technik.text_deploy2") }}
      />
      <div className="max-w-3xl w-full self-center px-2 md:px-0">
        <Disclaimer />
      </div>
    </div>
  );
}
