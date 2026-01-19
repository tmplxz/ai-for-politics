import { useState } from "react";
import { Button } from "./ui/button";
import { useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "./ui/dialog";

export default function MoreInfoButton({ card, images_dir, reasoningData, visualPoints, setIsHovered }) {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  const turnOff = visualPoints == "No data for selected party." ? true : false;

  const imageIndexes = [0, 1, 2, 3, 4];
  const { t } = useTranslation();

  return (
    <Dialog className="overflow-y-auto">
      <DialogTrigger asChild>
        <Button onClick={() => setIsHovered(false)} className="bg-primary text-white" disabled={turnOff}>
          Details
        </Button>
      </DialogTrigger>

      <DialogContent 
        showCloseButton={false} 
        className="xs:pt-12 sm:max-w-4xl md:max-w-[63rem] text-white border-none"
        closeClassName="bg-secondary text-white hover:bg-secondary/90 rounded-full"
      >

        <DialogClose asChild>
          <button className="absolute right-4 top-4 bg-secondary text-white rounded-full p-2 hover:bg-secondary/90">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M18 6 6 18"></path>
              <path d="m6 6 12 12"></path>
            </svg>
            <span className="sr-only">Close</span>
          </button>
        </DialogClose>

        <DialogHeader className="xs:fixed xs:pt-6">
          <DialogTitle className="text-2xl">{card.name}</DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 py-4 items-start">
          {/* Left Column: Description */}
          <div className="max-h-[32rem] overflow-y-auto pr-2 border border-secondary pb-4 rounded-lg p-3">
            <h3 className="font-semibold mb-2 text-lg">{t("details.title")}</h3>
            <DialogDescription className="text-gray-300 mb-4">
              {t("details.text")}
            </DialogDescription>
            <DialogDescription className="text-gray-300 mb-4">
              {reasoningData}
            </DialogDescription>
            <DialogDescription className="text-gray-300 mb-4">
              {t("details.closing1")} <a href={card.url} target="_blank">Website</a> {t("details.closing2")}
            </DialogDescription>
          </div>

          {/* Right Column: Main Image + Thumbnails */}
          <div className="flex flex-col gap-4">
            <div className="relative aspect-video w-full">
              <img
                className="h-full w-full object-contain rounded-lg"
                src={`${images_dir}/0_${selectedImageIndex}.png`}
                alt={`${card.name} - Image ${selectedImageIndex + 1}`}
                key={selectedImageIndex}
              />
            </div>
            <div className="flex gap-2 justify-center md:justify-start">
              {imageIndexes.map((index) => (
                <div
                  key={index}
                  className={`relative aspect-square w-16 h-16 md:w-24 md:h-24 cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
                    selectedImageIndex === index
                      ? "border-secondary scale-105"
                      : "border-transparent hover:border-gray-500"
                  }`}
                  onClick={() => setSelectedImageIndex(index)}
                >
                  <img
                    className="h-full w-full object-cover"
                    src={`${images_dir}/0_${index}.png`}
                    alt={`${images_dir} thumbnail ${index + 1}`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
        <DialogFooter className="flex justify-end gap-4">
          <DialogClose asChild>
            <Button onClick={() => setIsHovered(false)} className="bg-secondary">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
