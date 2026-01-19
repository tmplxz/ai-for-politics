import Disclaimer from "../components/Disclaimer";
import { useTranslation } from "react-i18next";

const base = import.meta.env.BASE_URL;

export default function ProjectPage() {
  const { t } = useTranslation();
  const members = [
    {
      name: "Raphael Fischer",
      img: "members/raphael.jpg",
      link: "https://www.linkedin.com/in/raphael-fischer-3b1046208/",
    },
    {
      name: "Nico Koltermann",
      img: "members/nico.jpg",
      link: "https://www.linkedin.com/in/nico-koltermann/",
    },
    {
      name: "Jan Krawiec",
      img: "members/jan.jpg",
      link: "https://www.linkedin.com/in/jan-krawiec-707515296/",
    },
    {
      name: "Louisa von Essen",
      img: "members/louisa.jpg",
      link: "https://www.linkedin.com/in/louisa-von-essen-a44b1a192/",
    },
    {
      name: "Youssef Abdelrahim",
      img: "members/youssef.jpg",
      link: "https://www.linkedin.com/in/youssef-abdelrahim-de/",
    },
    {
      name: "Tareq Khouja",
      img: "members/tareq.jpg",
      link: "https://www.linkedin.com/in/tareq-khouja/",
    },
  ];

  return (
    // general project goal and yail info
    <div className="flex flex-col items-center justify-center px-4 md:px-0">
      <div
        className="w-full lg:max-w-[50%] my-2"
        dangerouslySetInnerHTML={{ __html: t("projekt.text_top1") }}
      />
      <div className="max-w-3xl w-full self-center px-2 md:px-0">
        <Disclaimer />
      </div>
    </div>
  );
}
