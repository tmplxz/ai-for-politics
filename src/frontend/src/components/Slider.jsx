import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "./ui/carousel";
import Autoplay from "embla-carousel-autoplay";
import { usePartyImages } from "../../hooks/usePartyImages";
import { FaCheckCircle } from "react-icons/fa";
import FadeLoader from "react-spinners/FadeLoader";
import { useTranslation } from "react-i18next";

export function Slider({ metadata, lang = "de" }) {
  const { t } = useTranslation();

  const { images, isLoading } = usePartyImages(metadata, lang, true);

    if (isLoading)
      return (
        <div className="flex justify-center items-center h-full min-h-[400px]">
          <FadeLoader color="#52B49B" />
        </div>
      );

  return (
    <Carousel
      className="w-full"
      opts={{
        loop: true,
      }}
      plugins={[
        Autoplay({
          delay: 6000,
        }),
      ]}
    >
      <CarouselContent>
        {images.map((image) => (
          <CarouselItem key={image.id}>
            <div className="flex xs:flex-col sm:flex-row p-5 gap-24 items-center justify-center">
              <div className="max-w-[400px] text-left">
                <h2 className="text-2xl mb-5">
                  {t("home.slider_title")}{" "}
                  <a target="_blank" href={image.partyURL}>
                    {image.partyName}
                  </a>
                  ?
                </h2>
                <div className="flex flex-col gap-5">
                  {image.visualImpactPoints.map((point, idx) => (
                    <div key={idx} className="flex items-start">
                      <FaCheckCircle className="mr-2 mt-2 h-5 w-5 flex-shrink-0 self-start text-secondary" />
                      <span className="text-xl text-gray-700">
                        {point.charAt(0).toUpperCase() + point.slice(1)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg overflow-hidden shadow-lg max-w-[390px] aspect-square">
                <img
                  src={image.src}
                  alt={image.alt}
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious className="bg-transparent text-green-600 md:bg-secondary md:text-white" />
      <CarouselNext className="bg-transparent text-green-600 md:bg-secondary md:text-white" />
    </Carousel>
  );
}
