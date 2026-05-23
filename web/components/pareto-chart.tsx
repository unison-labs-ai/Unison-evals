"use client";

import { useState } from "react";

export interface ParetoPoint {
  system: string;
  x: number;
  y: number;
}

interface ParetoChartProps {
  points: ParetoPoint[];
  xLabel: string;
  yLabel: string;
  width?: number;
  height?: number;
  invertX?: boolean;
}

const PALETTE = [
  "#4ade80",
  "#60a5fa",
  "#f59e0b",
  "#f87171",
  "#a78bfa",
  "#34d399",
  "#fb923c",
  "#38bdf8",
];

const MARGIN = { top: 20, right: 20, bottom: 48, left: 56 };

function domainFromPoints(
  points: ParetoPoint[],
  key: "x" | "y"
): [number, number] {
  if (points.length === 0) return [0, 1];
  const vals = points.map((p) => p[key]);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (min === max) return [min * 0.9, max * 1.1 || 1];
  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

function computePareto(
  points: ParetoPoint[],
  invertX: boolean
): ParetoPoint[] {
  // A point is on the Pareto frontier if no other point is better on BOTH axes.
  // "Better" = higher y AND (lower x if invertX else higher x).
  const dominated = new Set<number>();
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const pi = points[i];
      const pj = points[j];
      const jBetterX = invertX ? pj.x <= pi.x : pj.x >= pi.x;
      const jBetterY = pj.y >= pi.y;
      const jStrictlyBetterSomething = invertX
        ? pj.x < pi.x || pj.y > pi.y
        : pj.x > pi.x || pj.y > pi.y;
      if (jBetterX && jBetterY && jStrictlyBetterSomething) {
        dominated.add(i);
        break;
      }
    }
  }
  return points.filter((_, i) => !dominated.has(i));
}

export function ParetoChart({
  points,
  xLabel,
  yLabel,
  width = 600,
  height = 300,
  invertX = false,
}: ParetoChartProps) {
  const [hovered, setHovered] = useState<string | null>(null);

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const [xMin, xMax] = domainFromPoints(points, "x");
  const [yMin, yMax] = domainFromPoints(points, "y");

  function scaleX(v: number) {
    return ((v - xMin) / (xMax - xMin)) * plotW;
  }
  function scaleY(v: number) {
    return plotH - ((v - yMin) / (yMax - yMin)) * plotH;
  }

  const paretoPoints = computePareto(points, invertX);
  const paretoSorted = [...paretoPoints].sort((a, b) => a.x - b.x);

  const xTicks = 5;
  const yTicks = 5;

  function fmtNum(v: number) {
    if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(1);
  }

  return (
    <svg
      width={width}
      height={height}
      style={{ overflow: "visible", display: "block", maxWidth: "100%" }}
    >
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {/* Grid lines */}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const v = yMin + ((yMax - yMin) * i) / yTicks;
          const y = scaleY(v);
          return (
            <line
              key={i}
              x1={0}
              x2={plotW}
              y1={y}
              y2={y}
              stroke="var(--border)"
              strokeWidth={1}
            />
          );
        })}
        {Array.from({ length: xTicks + 1 }).map((_, i) => {
          const v = xMin + ((xMax - xMin) * i) / xTicks;
          const x = scaleX(v);
          return (
            <line
              key={i}
              x1={x}
              x2={x}
              y1={0}
              y2={plotH}
              stroke="var(--border)"
              strokeWidth={1}
            />
          );
        })}

        {/* X axis */}
        <line
          x1={0}
          x2={plotW}
          y1={plotH}
          y2={plotH}
          stroke="var(--muted)"
          strokeWidth={1}
        />
        {Array.from({ length: xTicks + 1 }).map((_, i) => {
          const v = xMin + ((xMax - xMin) * i) / xTicks;
          const x = scaleX(v);
          return (
            <g key={i}>
              <line
                x1={x}
                x2={x}
                y1={plotH}
                y2={plotH + 4}
                stroke="var(--muted)"
                strokeWidth={1}
              />
              <text
                x={x}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--muted)"
              >
                {fmtNum(v)}
              </text>
            </g>
          );
        })}
        <text
          x={plotW / 2}
          y={plotH + 36}
          textAnchor="middle"
          fontSize={11}
          fill="var(--muted)"
        >
          {xLabel}
        </text>

        {/* Y axis */}
        <line
          x1={0}
          x2={0}
          y1={0}
          y2={plotH}
          stroke="var(--muted)"
          strokeWidth={1}
        />
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const v = yMin + ((yMax - yMin) * i) / yTicks;
          const y = scaleY(v);
          return (
            <g key={i}>
              <line
                x1={-4}
                x2={0}
                y1={y}
                y2={y}
                stroke="var(--muted)"
                strokeWidth={1}
              />
              <text
                x={-8}
                y={y + 4}
                textAnchor="end"
                fontSize={10}
                fill="var(--muted)"
              >
                {fmtNum(v)}
              </text>
            </g>
          );
        })}
        <text
          x={0}
          y={-8}
          textAnchor="start"
          fontSize={11}
          fill="var(--muted)"
        >
          {yLabel}
        </text>

        {/* Pareto frontier */}
        {paretoSorted.length > 1 && (
          <polyline
            points={paretoSorted
              .map((p) => `${scaleX(p.x)},${scaleY(p.y)}`)
              .join(" ")}
            fill="none"
            stroke="var(--muted)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            strokeOpacity={0.6}
          />
        )}

        {/* Data points */}
        {points.map((p, i) => {
          const cx = scaleX(p.x);
          const cy = scaleY(p.y);
          const color = PALETTE[i % PALETTE.length];
          const isHovered = hovered === p.system;
          const onPareto = paretoPoints.some((pp) => pp.system === p.system);

          return (
            <g
              key={p.system}
              onMouseEnter={() => setHovered(p.system)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "default" }}
            >
              <circle
                cx={cx}
                cy={cy}
                r={isHovered ? 8 : 6}
                fill={color}
                fillOpacity={onPareto ? 0.9 : 0.55}
                stroke={color}
                strokeWidth={onPareto ? 2 : 1}
              />
              {/* Tooltip */}
              {isHovered && (
                <g>
                  <rect
                    x={cx + 10}
                    y={cy - 32}
                    width={160}
                    height={42}
                    rx={3}
                    fill="var(--card)"
                    stroke="var(--border)"
                    strokeWidth={1}
                  />
                  <text
                    x={cx + 18}
                    y={cy - 16}
                    fontSize={11}
                    fill="var(--fg)"
                    fontWeight={500}
                  >
                    {p.system}
                  </text>
                  <text x={cx + 18} y={cy - 2} fontSize={10} fill="var(--muted)">
                    {xLabel}: {fmtNum(p.x)} · {yLabel}: {fmtNum(p.y)}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
